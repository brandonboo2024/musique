"""Convert pilot samples of newer multi-hop datasets into MuSiQue's row format.

Pilot only: a few questions per dataset, to judge whether a dataset is harder
than MuSiQue for DeepSeek before committing to it.

  phantomwiki  PhantomWiki v1 (Gong et al., 2025), corpus size 50, seed 1.
               Synthetic, so it cannot be memorised. Every article is given
               as a paragraph. Answers are sets of names, dates or counts.
  morehopqa    MoreHopQA (Schnitzler et al., 2024), human-verified split.
               MuSiQue/HotpotQA/2Wiki questions with an extra commonsense,
               arithmetic or symbolic hop on top.
  multihoprag  MultiHop-RAG (Tang and Yang, COLM 2024), news articles from late
               2023. Its corpus (609 articles, ~1.5M tokens) needs retrieval, so
               each question gets its evidence articles plus distractors from the
               same category, eight articles in all, in a seeded shuffle.

Sources (Hugging Face, downloaded to ../../data/raw/hop_candidates/):
  kilian-group/phantom-wiki-v1  question-answer/ and text-corpus/ depth_20_size_50_seed_1
  alabnii/morehopqa             data/with_human_verification.json
  yixuantt/MultiHopRAG          MultiHopRAG.json, corpus.json (saved as mhrag_q.json, mhrag_corpus.json)

Needs pandas and pyarrow (run inside `nix develop .#cope`):
    python -m llm_prompting.pilot_data ../../data/raw/hop_candidates
"""

import ast
import json
import random
import sys
from pathlib import Path

SEED = 20261009
OUT = Path("data")


def phantomwiki(src, per_level=2, levels=(2, 5, 8, 11)):
    import pandas as pd
    qa = pd.read_parquet(src / "pw_qa_50.parquet")
    corpus = pd.read_parquet(src / "pw_corpus_50.parquet")
    paragraphs = [{"idx": i, "title": r.title, "paragraph_text": r.article,
                   "is_supporting": False} for i, r in enumerate(corpus.itertuples())]
    rng = random.Random(SEED)
    rows = []
    for level in levels:
        pool = sorted(qa[qa.difficulty.astype(int) == level].id)
        for qid in sorted(rng.sample(pool, per_level)):
            r = qa[qa.id == qid].iloc[0]
            answers = [str(a) for a in r.answer]
            rows.append({"id": f"pw_d{level}__{qid}", "paragraphs": paragraphs,
                         "question": r.question + " If there are several answers, "
                         "list all of them separated by commas.",
                         "answer": answers[0], "answer_aliases": answers[1:],
                         "answer_set": answers, "steps": level, "answerable": True})
    return rows


def morehopqa(src, per_hops=4, hop_counts=(3, 5)):
    data = json.loads((src / "morehop.json").read_text())
    rng = random.Random(SEED)
    rows = []
    for hops in hop_counts:
        pool = sorted(x["_id"] for x in data if int(x["no_of_hops"]) == hops)
        for qid in sorted(rng.sample(pool, per_hops)):
            x = next(x for x in data if x["_id"] == qid)
            context = x["context"] if isinstance(x["context"], list) else ast.literal_eval(x["context"])
            support = {s["paragraph_support_title"] for s in ast.literal_eval(
                str(x["question_decomposition"]))}
            paragraphs = [{"idx": i, "title": t, "paragraph_text": "".join(s),
                           "is_supporting": t in support} for i, (t, s) in enumerate(context)]
            rows.append({"id": f"mh_{hops}hop__{qid}", "paragraphs": paragraphs,
                         "question": x["question"], "answer": str(x["answer"]),
                         "answer_aliases": [], "steps": hops, "answerable": True})
    return rows


def multihoprag(src, per_type=2, articles=8):
    queries = json.loads((src / "mhrag_q.json").read_text())
    corpus = json.loads((src / "mhrag_corpus.json").read_text())
    rng = random.Random(SEED)
    rows = []
    for kind in ("inference_query", "comparison_query", "temporal_query", "null_query"):
        pool = [i for i, x in enumerate(queries) if x["question_type"] == kind]
        for i in sorted(rng.sample(pool, per_type)):
            x = queries[i]
            evidence = {e["title"] for e in x["evidence_list"]}
            gold = [a for a in corpus if a["title"] in evidence]
            category = gold[0]["category"] if gold else rng.choice(corpus)["category"]
            others = [a for a in corpus if a["category"] == category and a["title"] not in evidence]
            chosen = gold + rng.sample(others, articles - len(gold))
            rng.shuffle(chosen)
            paragraphs = [{"idx": j, "title": a["title"],
                           "paragraph_text": f"{a['source']}, {a['published_at'][:10]}. {a['body']}",
                           "is_supporting": a["title"] in evidence} for j, a in enumerate(chosen)]
            null = kind == "null_query"  # gold "Insufficient information." = our UNKNOWN
            rows.append({"id": f"mr_{kind.split('_')[0]}__{i}", "paragraphs": paragraphs,
                         "question": x["query"], "answer": "UNKNOWN" if null else x["answer"],
                         "answer_aliases": [], "steps": len(evidence), "answerable": True})
    return rows


def main():
    src = Path(sys.argv[1])
    for name, rows in (("phantomwiki", phantomwiki(src)), ("morehopqa", morehopqa(src)),
                       ("multihoprag", multihoprag(src))):
        path = OUT / f"pilot_{name}.jsonl"
        with path.open("w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(rows)} questions -> {path}")


if __name__ == "__main__":
    main()
