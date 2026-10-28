"""PhantomWiki sample: 48 questions in MuSiQue's row format, with the gold chain.

PhantomWiki v1 (Gong et al., 2025; github.com/kilian-group/phantom-wiki) is a
synthetic universe of invented people, so no model can have memorised it. We
use the released instance depth_20_size_50_seed_1 from Hugging Face
(kilian-group/phantom-wiki-v1): 51 articles (~8k tokens), all given to every
call as paragraphs, like MuSiQue's 20 paragraphs.

Sample: 8 questions at each difficulty 3, 6, 9, 12, 15, 18 (PhantomWiki's
count of reasoning steps), seed 20261009.

Gold chain: each question comes with a Prolog query, one goal per relation.
Running the goals from the named anchor outwards in SWI-Prolog, over the
instance's own fact and rule file, gives the gold answer set after every
relation. It is for tracing only and never reaches a model.

Scoring follows PhantomWiki's phantom_eval/score.py: split on commas, strip,
lower-case, compare as sets. EM = sets equal; F1 = set precision and recall.

    # inputs in ../../data/raw/hop_candidates/: pw_qa_50.parquet, pw_corpus_50.parquet,
    # pw_db_50.parquet (question-answer/, text-corpus/, database/ of the instance)
    nix develop ../..#cope -c ../cope/venv/bin/python -m llm_prompting.phantomwiki ../../data/raw/hop_candidates
    # needs swipl on PATH, e.g. inside `nix shell nixpkgs#swi-prolog`
"""

import json
import random
import re
import subprocess
import sys
from pathlib import Path

SAMPLE = Path("llm_prompting/sample_phantomwiki48.json")
GOLD = Path("data/cs470_phantomwiki48.jsonl")
DB = Path("data/phantomwiki_size50_seed1.pl")
SEED = 20261009
LEVELS = (3, 6, 9, 12, 15, 18)
PER_LEVEL = 8
MAX_STEPS = 12  # plan length cap; above the longest gold chain in the sample (checked on build)
NOTE = " If there are several answers, list all of them separated by commas."
SEP = ","


def normalize(answer):
    """phantom_eval.utils.normalize_pred."""
    return set(map(str.lower, map(str.strip, answer.split(SEP))))


def score(pred, gold):
    """phantom_eval.score exact_match and f1, with gold as a list of answers."""
    p, g = normalize(pred), normalize(SEP.join(gold))
    precision = sum(x in g for x in p) / len(p)
    recall = sum(x in p for x in g) / len(g)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"em": float(p == g), "f1": f1}


def chain_goals(query):
    """Goals ordered anchor first, each with the variable it introduces."""
    goals = list(reversed(list(query)))
    seen, out = set(), []
    for goal in goals:
        names = re.findall(r"\b([A-Z][A-Za-z]*_\d+)\b", goal)
        new = [v for v in names if v not in seen]
        target = "Count_1" if goal.startswith("aggregate_all") else new[-1]
        seen.update(names)
        out.append((goal, target))
    return out


ATTR_WORDS = {"dob": "date of birth", "job": "occupation", "hobby": "hobby"}


def gold_plan(row):
    """The gold chain as a plan in the planners' format: one subquestion per goal,
    #k for the answer of step k, worded as the question words each relation."""
    question = row["question"].lower()
    introduced, plan = {}, []  # variable -> step number that introduced it

    def subject(arg):
        return arg.strip('"') if arg.startswith('"') else f"#{introduced[arg]}"

    def wording(rel):
        words = rel.replace("_", " ")
        for form in (words.replace(" in law", "-in-law").replace("great ", "great-"), words):
            if form in question:
                return form
        raise ValueError(f"{rel!r} not worded in {row['question']!r}")

    for goal in (g["goal"] for g in row["gold_chain"]):
        if m := re.match(r"aggregate_all\(count, distinct\((\w+)\((\w+), \w+\)\), \w+\)$", goal):
            noun = re.search(r"how many ([\w\- ]+?) does", question)[1]
            plan.append(f"How many {noun} does {subject(m[2])} have?")
            continue
        pred, a, b = re.match(r"(\w+)\(([^,]+), (.+)\)$", goal).groups()
        if pred in ATTR_WORDS and b.startswith('"'):
            plan.append(f"Who is the person whose {ATTR_WORDS[pred]} is {b.strip(chr(34))}?")
        elif pred in ATTR_WORDS:
            plan.append(f"What is the {ATTR_WORDS[pred]} of {subject(a)}?")
        else:
            plan.append(f"Who is the {wording(pred)} of {subject(a)}?")
        introduced[b if pred not in ATTR_WORDS or not b.startswith('"') else a] = len(plan)
    return plan


def gold_chains(rows):
    """Run every prefix of every question's chain in SWI-Prolog."""
    lines = [f':- consult("{DB}").', ":- initialization(main, main).",
             "main :- forall(step(Id, I, V, G), (findall(V, G, L), sort(L, S),"
             ' format("~w\\t~w\\t~q~n", [Id, I, S]))).']
    for qid, query in rows:
        goals = chain_goals(query)
        for i in range(len(goals)):
            body = ", ".join(g for g, _ in goals[: i + 1])
            lines.append(f'step("{qid}", {i}, {goals[i][1]}, ({body})).')
    program = Path("data/phantomwiki_chains.pl")
    program.write_text("\n".join(lines) + "\n")
    out = subprocess.run(["swipl", str(program)], capture_output=True, text=True, check=True)
    chains = {}
    for line in out.stdout.splitlines():
        qid, i, values = line.split("\t")
        chains.setdefault(qid, {})[int(i)] = [str(v) for v in json.loads(values)]
    return chains


def main():
    import pandas as pd
    src = Path(sys.argv[1])
    qa = pd.read_parquet(src / "pw_qa_50.parquet")
    corpus = pd.read_parquet(src / "pw_corpus_50.parquet")
    DB.write_text(pd.read_parquet(src / "pw_db_50.parquet").content[0])
    if SAMPLE.exists():
        ids = json.loads(SAMPLE.read_text())["ids"]  # never redraw a committed sample
    else:
        rng = random.Random(SEED)
        ids = []
        for level in LEVELS:
            pool = sorted(qa[qa.difficulty.astype(int) == level].id)
            ids += sorted(rng.sample(pool, PER_LEVEL))
        SAMPLE.write_text(json.dumps(
            {"source": "kilian-group/phantom-wiki-v1 depth_20_size_50_seed_1",
             "seed": SEED, "levels": LEVELS, "ids": ids}, indent=1))
    by_id = {r.id: r for r in qa.itertuples()}
    chains = gold_chains([(i, by_id[i].prolog["query"]) for i in ids])
    paragraphs = [{"idx": i, "title": r.title, "paragraph_text": r.article, "is_supporting": False}
                  for i, r in enumerate(corpus.itertuples())]
    with GOLD.open("w") as out:
        for qid in ids:
            r = by_id[qid]
            answers = [str(a) for a in r.answer]
            goals = chain_goals(r.prolog["query"])
            chain = [{"goal": g, "answers": chains[qid][i]} for i, (g, _) in enumerate(goals)]
            assert set(chain[-1]["answers"]) == set(answers), (qid, chain[-1], answers)
            assert len(chain) <= MAX_STEPS, (qid, len(chain))
            out.write(json.dumps({
                "id": qid, "paragraphs": paragraphs, "question": r.question + NOTE,
                "answer": answers[0], "answer_aliases": answers[1:], "answer_set": answers,
                "answerable": True, "difficulty": int(r.difficulty), "type": int(r.type),
                "gold_chain": chain, "max_steps": MAX_STEPS, "answer_note": NOTE,
            }, ensure_ascii=False) + "\n")
    print(f"{len(ids)} questions -> {SAMPLE}, {GOLD}")


if __name__ == "__main__":
    main()
