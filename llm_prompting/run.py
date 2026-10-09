"""Run one method for one repetition on the 48-question dev sample.

Methods (who plans -> who answers):
  small          -        -> Qwen3-4B      (one call)
  deepseek       -        -> DeepSeek      (one call)
  deepseek_plan  DeepSeek -> Qwen3-4B      (plan once per question, then one call per hop)
  small_plan     Qwen3-4B -> Qwen3-4B      (same, Qwen plans)

Plans are written once per question and reused unchanged across repetitions.
The final answer of a planned method is the answer to its last hop.

Outputs, per method and repetition:
  predictions/cs470/<method>_rep<N>.jsonl          official format, for evaluate_v1.0.py
  predictions/cs470/<method>_rep<N>.details.jsonl  plan, hop answers, tokens, cost per question
  logs/cs470/...calls.jsonl                        every model call

    python -m llm_prompting.run --method small --rep 0 [--limit 2]
    python -m llm_prompting.run --evaluate
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from llm_prompting import prompts
from llm_prompting.models import CallLog, DeepSeek, Qwen

GOLD = Path("data/cs470_dev48.jsonl")
PRED = Path("predictions/cs470")
LOGS = Path("logs/cs470")
METHODS = ("small", "deepseek", "deepseek_plan", "small_plan")
DIRECT_CAP, PLAN_CAP, HOP_CAP = 4096, 512, 640  # max output tokens, as in Phase 2b

_models = {}


def model(name):
    if name not in _models:
        _models[name] = Qwen() if name == "qwen" else DeepSeek()
    return _models[name]


def cost(calls):
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "usd": round(sum(c["usd"] for c in calls), 6),
        "capped_calls": sum(c["finish_reason"] == "length" for c in calls),
    }


def answer_directly(row, who, log, rep):
    call = log.call(model(who), f"{row['id']}/rep{rep}/direct",
                    prompts.direct_prompt(row), DIRECT_CAP)
    answer, valid = prompts.final_answer(call["text"])
    return answer, {"valid_format": valid, "cost": cost([call])}


def answer_with_plan(row, planner, plan_log, log, rep):
    plan_call = plan_log.call(model(planner), f"{row['id']}/plan",
                              prompts.plan_prompt(row), PLAN_CAP)
    steps = prompts.parse_plan(plan_call["text"])
    calls = [plan_call]
    if steps is None:  # unusable plan: the question counts as wrong
        return "", {"plan": None, "raw_plan": plan_call["text"], "cost": cost(calls)}
    answers, hops = [], []
    for i, step in enumerate(steps):
        question = prompts.substitute(step, answers)
        call = log.call(model("qwen"), f"{row['id']}/rep{rep}/hop{i}",
                        prompts.hop_prompt(question, row), HOP_CAP)
        answer, valid = prompts.final_answer(call["text"])
        answers.append(answer)
        hops.append({"question": question, "answer": answer, "valid_format": valid})
        calls.append(call)
    return answers[-1], {"plan": steps, "hops": hops, "cost": cost(calls),
                         "plan_cost": cost([plan_call])}


def run(method, rep, limit):
    rows = [json.loads(line) for line in GOLD.open()][:limit]
    log = CallLog(LOGS / f"{method}_rep{rep}.calls.jsonl")
    planner = {"deepseek_plan": "deepseek", "small_plan": "qwen"}.get(method)
    plan_log = CallLog(LOGS / f"plans_{planner}.calls.jsonl") if planner else None
    PRED.mkdir(parents=True, exist_ok=True)
    predictions, details = [], []
    for n, row in enumerate(rows, 1):
        if planner:
            answer, info = answer_with_plan(row, planner, plan_log, log, rep)
        else:
            answer, info = answer_directly(row, "qwen" if method == "small" else "deepseek",
                                           log, rep)
        predictions.append({"id": row["id"], "predicted_answer": answer,
                            "predicted_support_idxs": [], "predicted_answerable": True})
        details.append({"id": row["id"], "method": method, "rep": rep,
                        "answer": answer, "gold": row["answer"], **info})
        print(f"{method} rep{rep} {n}/{len(rows)} {row['id']}: {answer!r} (gold {row['answer']!r})")
    suffix = "" if limit is None else f".limit{limit}"
    write_jsonl(PRED / f"{method}_rep{rep}{suffix}.jsonl", predictions)
    write_jsonl(PRED / f"{method}_rep{rep}{suffix}.details.jsonl", details)


def write_jsonl(path, rows):
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate():
    """Score every complete prediction file with the official evaluator."""
    for path in sorted(PRED.glob("*_rep[0-9].jsonl")):
        metrics = path.with_suffix(".metrics.json")
        subprocess.run([sys.executable, "evaluate_v1.0.py", str(path), str(GOLD),
                        "--output_filepath", str(metrics)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--limit", type=int, help="smoke test on the first N questions")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        run(args.method, args.rep, args.limit)


if __name__ == "__main__":
    main()
