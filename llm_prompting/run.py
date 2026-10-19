"""Run one method for one repetition on the 48-question dev sample.

Methods (who plans -> who answers):
  small          -        -> Qwen3-4B      (one call)
  deepseek       -        -> DeepSeek      (one call)
  deepseek_plan  DeepSeek -> Qwen3-4B      (plan once per question, then one call per hop)
  small_plan     Qwen3-4B -> Qwen3-4B      (same, Qwen plans)
  deepseek_plan_deepseek  DeepSeek -> DeepSeek  (the same DeepSeek plans, DeepSeek answers)

Plans are written once per question and reused unchanged across repetitions.

--small qwen3-8b|gemma3-4b replaces Qwen3-4B with another small model served by a
local vLLM server (models.SMALL_MODELS). Its outputs go to a <small>/ subdirectory,
and the DeepSeek plans are the saved ones: a plan missing from the log is an error,
never a new DeepSeek call.
The final answer of a planned method is the answer to its last hop.

Outputs, per method and repetition:
  predictions/cs470/<method>_rep<N>.jsonl          official format, for evaluate_v1.0.py
  predictions/cs470/<method>_rep<N>.details.jsonl  plan, hop answers, tokens, cost per question
  logs/cs470/...calls.jsonl                        every model call

    python -m llm_prompting.run --method small --rep 0 [--limit 2]
    python -m llm_prompting.run --dataset phantomwiki48 --method deepseek --rep 5 --only 9,17,41
    python -m llm_prompting.run --evaluate
    python -m llm_prompting.run --small qwen3-8b --dataset phantomwiki48 --method deepseek_plan --rep 0
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from llm_prompting import prompts
from llm_prompting.models import SMALL_MODELS, CallLog, DeepSeek, LocalSmall, Qwen

GOLD = Path("data/cs470_dev48.jsonl")
PRED = Path("predictions/cs470")
LOGS = Path("logs/cs470")
PILOTS = ("phantomwiki", "morehopqa", "multihoprag")  # see pilot_data.py; outputs go to <dir>/pilot_<name>/
DATASETS = ("phantomwiki48",) + PILOTS
METHODS = ("small", "deepseek", "deepseek_plan", "small_plan", "deepseek_plan_deepseek")
# method -> (planner, answerer) for the planned methods
PLANNED = {"deepseek_plan": ("deepseek", "qwen"), "small_plan": ("qwen", "qwen"),
           "deepseek_plan_deepseek": ("deepseek", "deepseek")}
DIRECT_CAP, PLAN_CAP, HOP_CAP = 4096, 512, 640  # max output tokens, as in Phase 2b

SMALL = None  # None: Qwen3-4B in BF16 (Phase 2); else a models.SMALL_MODELS name
_models = {}


def model(name):
    if name not in _models:
        if name == "qwen":
            _models[name] = LocalSmall(SMALL) if SMALL else Qwen()
        else:
            _models[name] = DeepSeek()
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
    return answer, {"valid_format": valid, "finish_reason": call["finish_reason"],
                    "cost": cost([call])}


def answer_with_plan(row, planner, answerer, plan_log, log, rep):
    key = f"{row['id']}/plan"
    if SMALL and planner == "deepseek":
        assert key in plan_log.done, f"no saved DeepSeek plan for {key}"
        assert plan_log.done[key]["prompt_sha256"] == hashlib.sha256(json.dumps(
            prompts.plan_prompt(row), sort_keys=True).encode()).hexdigest(), key
    plan_call = plan_log.call(model(planner), key, prompts.plan_prompt(row), PLAN_CAP)
    steps = prompts.parse_plan(plan_call["text"], row.get("max_steps", prompts.MAX_STEPS))
    calls = [plan_call]
    if steps is None:  # unusable plan: the question counts as wrong
        return "", {"plan": None, "raw_plan": plan_call["text"], "cost": cost(calls)}
    answers, hops = [], []
    for i, step in enumerate(steps):
        question = prompts.substitute(step, answers) + row.get("answer_note", "")
        call = log.call(model(answerer), f"{row['id']}/rep{rep}/hop{i}",
                        prompts.hop_prompt(question, row), HOP_CAP)
        answer, valid = prompts.final_answer(call["text"])
        answers.append(answer)
        hops.append({"question": question, "answer": answer, "valid_format": valid,
                     "finish_reason": call["finish_reason"],
                     "output_tokens": call["output_tokens"]})
        calls.append(call)
    return answers[-1], {"plan": steps, "hops": hops, "cost": cost(calls),
                         "plan_cost": cost([plan_call])}


def use_dataset(name):
    """Point GOLD, PRED and LOGS at another dataset instead of the MuSiQue sample."""
    global GOLD, PRED, LOGS
    sub = f"pilot_{name}" if name in PILOTS else name
    GOLD = Path(f"data/{sub}.jsonl" if name in PILOTS else f"data/cs470_{name}.jsonl")
    PRED, LOGS = PRED / sub, LOGS / sub


def run(method, rep, limit, only=None):
    rows = [json.loads(line) for line in GOLD.open()][:limit]
    if only:  # question numbers as in the trace (Q1 = first row)
        rows = [rows[n - 1] for n in only]
    planner, answerer = PLANNED.get(method, (None, None))
    pred, logs = PRED, LOGS
    if SMALL:
        assert "qwen" in (planner, answerer) or method == "small", \
            f"{method} uses no small model; reuse the saved run"
        pred, logs = PRED / SMALL, LOGS / SMALL
    log = CallLog(logs / f"{method}_rep{rep}.calls.jsonl")
    # DeepSeek's plans are shared by every small model; a small model's own plans are not.
    plan_log = CallLog((LOGS if planner == "deepseek" else logs)
                       / f"plans_{planner}.calls.jsonl") if planner else None
    pred.mkdir(parents=True, exist_ok=True)
    predictions, details = [], []
    for n, row in enumerate(rows, 1):
        if planner:
            answer, info = answer_with_plan(row, planner, answerer, plan_log, log, rep)
        else:
            answer, info = answer_directly(row, "qwen" if method == "small" else "deepseek",
                                           log, rep)
        predictions.append({"id": row["id"], "predicted_answer": answer,
                            "predicted_support_idxs": [], "predicted_answerable": True})
        details.append({"id": row["id"], "method": method, "rep": rep,
                        "answer": answer, "gold": row["answer"], **info})
        print(f"{method} rep{rep} {n}/{len(rows)} {row['id']}: {answer!r} (gold {row['answer']!r})")
    suffix = ("" if limit is None else f".limit{limit}") + (
        f".only{'-'.join(map(str, only))}" if only else "")
    write_jsonl(pred / f"{method}_rep{rep}{suffix}.jsonl", predictions)
    write_jsonl(pred / f"{method}_rep{rep}{suffix}.details.jsonl", details)


def write_jsonl(path, rows):
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate():
    """Score every complete prediction file with the official evaluator.

    PhantomWiki answers are sets, which MuSiQue's evaluator cannot score, so
    they are scored with PhantomWiki's own rule (phantomwiki.score)."""
    if GOLD.name == "cs470_phantomwiki48.jsonl":
        from llm_prompting.phantomwiki import score
        gold = {json.loads(line)["id"]: json.loads(line)["answer_set"] for line in GOLD.open()}
        for path in sorted(PRED.glob("*_rep[0-9].jsonl")):
            scores = {r["id"]: score(r["predicted_answer"], gold[r["id"]])
                      for r in map(json.loads, path.open())}
            n = len(scores)
            path.with_suffix(".metrics.json").write_text(json.dumps({
                "answer_em": sum(s["em"] for s in scores.values()) / n,
                "answer_f1": sum(s["f1"] for s in scores.values()) / n,
                "per_question": scores}, indent=1))
        return
    for path in sorted(PRED.glob("*_rep[0-9].jsonl")):
        metrics = path.with_suffix(".metrics.json")
        subprocess.run([sys.executable, "evaluate_v1.0.py", str(path), str(GOLD),
                        "--output_filepath", str(metrics)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--limit", type=int, help="smoke test on the first N questions")
    parser.add_argument("--only", type=lambda s: [int(n) for n in s.split(",")],
                        help="only these question numbers, as in the trace (e.g. 9,17,41)")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--small", choices=sorted(SMALL_MODELS),
                        help="another small model on the local vLLM server instead of Qwen3-4B")
    parser.add_argument("--dataset", choices=DATASETS, help="another dataset instead of MuSiQue")
    args = parser.parse_args()
    if args.dataset:
        use_dataset(args.dataset)
    global SMALL, PRED
    SMALL = args.small
    if args.evaluate:
        if SMALL:
            PRED = PRED / SMALL
        evaluate()
    else:
        run(args.method, args.rep, args.limit, args.only)


if __name__ == "__main__":
    main()
