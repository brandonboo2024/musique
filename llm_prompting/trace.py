"""Write an org file that traces every model call, question by question.

For each question: the gold chain (PhantomWiki's Prolog steps, or MuSiQue's
gold decomposition), then for each method the plan, every hop's exact input
and raw output, the answer read from it, and which gold step (if any) that
answer equals. A hop that matches no gold step is where a chain went wrong.

Prompts are rebuilt from the code and data exactly as run.py sends them; only
the passage list is elided, since it is identical in every call of a question
(see the corpus section at the end, or the dataset file).

    python -m llm_prompting.trace --dataset phantomwiki48 [--rep 0]
    python -m llm_prompting.trace            # MuSiQue dev48
    python -m llm_prompting.trace --dataset phantomwiki48 --rep 5 --suffix .only9-17-41
Writes predictions/cs470/<dataset>/trace_rep<N>.org.
"""

import argparse
import json
from pathlib import Path

from llm_prompting import prompts, run
from metrics.answer import normalize_answer

ELIDED = "<passages elided: the same {n} passages in every call for this question>"


def load_calls(path):
    calls = {}
    if path.exists():
        for line in path.open():
            record = json.loads(line)
            calls[record["key"]] = record
    return calls


def answer_set(text, multi):
    """Normalised answers; PhantomWiki answers are comma-separated sets."""
    parts = text.split(",") if multi else [text]
    return {normalize_answer(a) for a in parts if normalize_answer(a)}


def gold_steps(row):
    """[(label, [answers])] from PhantomWiki's chain or MuSiQue's decomposition."""
    if "gold_chain" in row:
        return [(c["goal"], c["answers"]) for c in row["gold_chain"]]
    return [(d["question"], [d["answer"]]) for d in row.get("question_decomposition", [])]


def matches(answer, steps, multi):
    found = answer_set(answer, multi)
    return [i + 1 for i, (_, gold) in enumerate(steps)
            if found and found == set().union(*(answer_set(g, multi) for g in gold))]


def correct(answer, row):
    if "answer_set" in row:
        from llm_prompting.phantomwiki import score
        return score(answer, row["answer_set"])
    gold = {normalize_answer(a) for a in [row["answer"], *row.get("answer_aliases", [])]}
    return {"em": float(normalize_answer(answer) in gold), "f1": None}


def first_break(d, steps, multi):
    """'h3/5': hop 3 of a 5-step plan is the first whose answer equals no gold step."""
    if not d.get("plan"):
        return "plan unusable"
    for i, hop in enumerate(d["hops"], 1):
        if not matches(hop["answer"], steps, multi):
            return f"h{i}/{len(d['plan'])}"
    return f"ok/{len(d['plan'])}"


def show_prompt(messages, row):
    """The messages as sent, with the passage list replaced by a placeholder."""
    out = []
    for m in messages:
        text = m["content"]
        if m["role"] == "user":
            head, _, instruction = text.partition("\n")
            payload = json.loads(head)
            payload["passages"] = ELIDED.format(n=len(payload["passages"]))
            text = json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n" + instruction
        out.append(f"[{m['role']}]\n{text}")
    return "\n\n".join(out)


def block(text):
    # A leading comma escapes lines org would read as headings or block ends.
    lines = [("," + l if l.startswith(("*", "#+")) else l) for l in text.splitlines()]
    return "#+begin_example\n" + "\n".join(lines) + "\n#+end_example\n"


def passage_block(row):
    return block("\n\n".join(
        f"[{p['idx']}]{'*' if p.get('is_supporting') else ''} {p['title']}\n{p['paragraph_text']}"
        for p in row["paragraphs"]))


def call_meta(call):
    return (f"{call['model']}, {call['input_tokens']} tokens in, {call['output_tokens']} out, "
            f"finish={call['finish_reason']}, {call['seconds']} s")


def trace(rep, suffix=""):
    rows = [json.loads(line) for line in run.GOLD.open()]
    methods = [m for m in run.METHODS
               if (run.PRED / f"{m}_rep{rep}{suffix}.details.jsonl").exists()]
    details = {m: {d["id"]: d for d in map(json.loads,
                   (run.PRED / f"{m}_rep{rep}{suffix}.details.jsonl").open())} for m in methods}
    calls = {m: load_calls(run.LOGS / f"{m}_rep{rep}.calls.jsonl") for m in methods}
    plans = {p: load_calls(run.LOGS / f"plans_{p}.calls.jsonl") for p in ("deepseek", "qwen")}

    out = [f"#+title: Trace of every model call, {run.GOLD.name}, rep {rep}",
           "#+startup: overview", "",
           "Each question: gold chain, then per method the plan, each hop's input and raw output.",
           "\"matches gold step k\" means the hop's answer set equals the gold answers after step k;",
           "the first hop that matches no gold step is usually where the chain broke.", "",
           "* Summary (EM per method; F1 in brackets)",
           "Planned methods also show the first hop whose answer equals no gold step, out of the",
           "plan's length (h2/5), or ok/5 if every hop equals some gold step. Gold chain length is in",
           "the 'chain' column.",
           "| # | id | difficulty | chain | " + " | ".join(methods) + " |", "|-"]
    body = []
    for n, row in enumerate(rows, 1):
        if not any(row["id"] in details[m] for m in methods):
            continue  # not in this (partial) run; Q numbers stay those of the full sample
        steps = gold_steps(row)
        level = row.get("difficulty", row["id"].split("hop")[0] + "hop")
        cells, sections = [], []
        for m in methods:
            d = details[m].get(row["id"])
            if d is None:
                cells.append("-")
                continue
            s = correct(d["answer"], row)
            cell = f"{s['em']:.0f}" + ("" if s["f1"] is None else f" ({s['f1']:.2f})")
            if m in run.PLANNED:
                cell += " " + first_break(d, steps, "answer_set" in row)
            cells.append(cell)
            sections.append(trace_method(m, d, row, steps, calls[m], plans, rep, s))
        out.append(f"| {n} | [[*Q{n}][{row['id'][:12]}]] | {level} | {len(steps)} | "
                   + " | ".join(cells) + " |")
        body.append(f"* Q{n}\n:PROPERTIES:\n:ID_FULL: {row['id']}\n:END:")
        body.append(f"*Question:* {row['question']}\n\n*Gold answer:* "
                    + ", ".join(row.get("answer_set") or [row["answer"], *row.get("answer_aliases", [])]))
        body.append("** Gold chain\n| step | relation / sub-question | gold answers |\n|-")
        body += [f"| {i} | ={label}= | {', '.join(gold)} |" for i, (label, gold) in enumerate(steps, 1)]
        body += sections
        if "gold_chain" not in row:  # MuSiQue: each question has its own 20 passages
            body.append("** Passages (gold-supporting ones marked *)\n" + passage_block(row))
    if "gold_chain" in rows[0]:  # PhantomWiki: one corpus shared by every question
        body.append("* Corpus (identical in every call)\n" + passage_block(rows[0]))
    path = run.PRED / f"trace_rep{rep}{suffix}.org"
    path.write_text("\n".join(out + body) + "\n")
    print(f"{len(rows)} questions, methods {methods} -> {path}")


def trace_method(method, d, row, steps, calls, plans, rep, s):
    score = f"EM {s['em']:.0f}" + ("" if s["f1"] is None else f", F1 {s['f1']:.2f}")
    lines = [f"** {method}: {d['answer'][:80]!r} ({score})"]
    if method not in run.PLANNED:
        call = calls[f"{row['id']}/rep{rep}/direct"]
        lines += [f"*** Input\n{block(show_prompt(prompts.direct_prompt(row), row))}",
                  f"*** Output ({call_meta(call)})\n{block(call['text'])}"]
        return "\n".join(lines)
    planner = run.PLANNED[method][0]
    plan = plans[planner][f"{row['id']}/plan"]
    lines.append(f"*** Plan by {planner}: {len(d['plan']) if d['plan'] else 'UNUSABLE'} steps "
                 f"(gold chain has {len(steps)})")
    if d["plan"]:
        lines += [f"{i}. {q}" for i, q in enumerate(d["plan"], 1)]
    lines.append(f"**** Plan call input and raw output ({call_meta(plan)})\n"
                 f"{block(show_prompt(prompts.plan_prompt(row), row))}{block(plan['text'])}")
    for i, hop in enumerate(d.get("hops", [])):
        call = calls[f"{row['id']}/rep{rep}/hop{i}"]
        hit = matches(hop["answer"], steps, "answer_set" in row)
        tag = f"matches gold step {', '.join(map(str, hit))}" if hit else "matches NO gold step"
        lines += [f"*** Hop {i + 1}: {hop['answer'][:70]!r} [{tag}]",
                  f"Sub-question sent: {hop['question']}",
                  f"**** Input\n{block(show_prompt(prompts.hop_prompt(hop['question'], row), row))}",
                  f"**** Output ({call_meta(call)})\n{block(call['text'])}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=run.DATASETS)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--suffix", default="",
                        help="trace a partial run, e.g. .only9-17-41 or .limit2")
    args = parser.parse_args()
    if args.dataset:
        run.use_dataset(args.dataset)
    trace(args.rep, args.suffix)


if __name__ == "__main__":
    main()
