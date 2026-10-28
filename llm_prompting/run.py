"""Run one method for one repetition on the 48-question dev sample.

Methods (who plans -> who answers):
  small          -        -> Qwen3-4B      (one call)
  deepseek       -        -> DeepSeek      (one call)
  deepseek_plan  DeepSeek -> Qwen3-4B      (plan once per question, then one call per hop)
  small_plan     Qwen3-4B -> Qwen3-4B      (same, Qwen plans)
  deepseek_plan_deepseek  DeepSeek -> DeepSeek  (the same DeepSeek plans, DeepSeek answers)

Plans are written once per question and reused unchanged across repetitions.

Each planned method also has two variants that end the chain with an aggregation
call (the original question, the passages and every subquestion with its answer)
instead of taking the last step's answer, so that a step that never received an
earlier answer (a plan without #k) cannot lose it:
  <method>_hidden   the steps are exactly the original method's (saved calls, reused
                    and checked by prompt hash); only the aggregation call is new
  <method>_context  every step also sees all earlier subquestions and their answers
Both reuse the saved plans and never draw a new one. For the methods whose steps
Qwen answers, <method>_hidden_dsread and <method>_context_dsread reuse the steps of
<method> and <method>_context and let DeepSeek make the aggregation call, which
separates the steps from the final reader.

--small <name> (qwen2.5-1.5b, qwen3-4b-orig, qwen3-8b, gemma3-4b, qwen3-4b-fp8) replaces
Qwen3-4B with another small model served by a local vLLM server
(models.SMALL_MODELS). Its outputs go to a <small>/ subdirectory, and the DeepSeek
plans are the saved ones: a plan missing from the log is an error, never a new
DeepSeek call. qwen3-4b-fp8 (Qwen3-4B-Instruct-2507 in FP8, for the 8 GB laptop) also
reuses the BF16 Qwen3-4B's saved plans instead of writing its own.
The final answer of a planned method is the answer to its last hop.

Planning as guidance, and the controls that separate the plan from the splitting:
  guided_deepseek_plan           DeepSeek's saved plan, Qwen answers the original
                                 question in one call with the plan in context
  guided_small_plan              the same with Qwen's saved plan
  guided_gold                    the same with the gold chain written as a plan
  guided_deepseek_plan_deepseek  DeepSeek's saved plan, DeepSeek answers in one call
  guided_gold_deepseek           the gold plan, DeepSeek answers in one call
  <guided method>_checked        the same, but the model is told the plan may be wrong
                                 and to check it against the question first
  <guided method>_sets           the same, but the model is told #k may stand for several
                                 people and to carry all of them
  <guided method>_commit         guidance with the no-abstain instruction of *_commit
  small_commit, deepseek_commit  direct, but told every question is answerable from the
                                 passages and to give the best-supported answer, not UNKNOWN
  guided_deepseek_rev[_deepseek] DeepSeek's saved plan after DeepSeek checks it against
                                 the question and corrects it (plans_deepseek_rev log)
  gold_plan                      the gold plan run as a chain, Qwen answers each step
  <method>_cap4096               deepseek_plan or small_plan with 4,096 output tokens
                                 per step (the direct cap) instead of 640; with
                                 _hidden, the final call over those steps
The guided methods use a saved plan only, never a new one, and its prompt must match.

Outputs, per method and repetition:
  predictions/cs470/<method>_rep<N>.jsonl          official format, for evaluate_v1.0.py
  predictions/cs470/<method>_rep<N>.details.jsonl  plan, hop answers, tokens, cost per question
  logs/cs470/...calls.jsonl                        every model call

    python -m llm_prompting.run --method small --rep 0 [--limit 2]
    python -m llm_prompting.run --dataset phantomwiki48 --method deepseek --rep 5 --only 9,17,41
    python -m llm_prompting.run --evaluate
    python -m llm_prompting.run --small qwen3-8b --dataset phantomwiki48 \
        --method deepseek_plan --rep 0
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from llm_prompting import phantomwiki, prompts, retrieval
from llm_prompting.models import (SHARES_QWEN_PLANS, SMALL_MODELS, CallLog, DeepSeek,
                                  LocalSmall, Qwen)

GOLD = Path("data/cs470_dev48.jsonl")
PRED = Path("predictions/cs470")
LOGS = Path("logs/cs470")
# see pilot_data.py; outputs go to <dir>/pilot_<name>/
PILOTS = ("phantomwiki", "morehopqa", "multihoprag")
DATASETS = ("phantomwiki48", "dev300", "phantomwiki146") + PILOTS
# method -> (planner, answerer) for the planned methods
PLANNED = {"deepseek_plan": ("deepseek", "qwen"), "small_plan": ("qwen", "qwen"),
           "deepseek_plan_deepseek": ("deepseek", "deepseek")}
MODES = ("hidden", "context")  # see the module docstring; "chain" is the original
DSREAD = "_dsread"  # suffix: DeepSeek makes the aggregation call over reused steps
CAP = "_cap4096"  # suffix: steps get the direct cap instead of HOP_CAP
PLANNED.update({f"{m}_{mode}": PLANNED[m] for m in list(PLANNED) for mode in MODES})
PLANNED.update({f"{m}_{mode}{DSREAD}": PLANNED[m] for m in ("deepseek_plan", "small_plan")
                for mode in MODES})
PLANNED["gold_plan"] = ("gold", "qwen")  # "gold": the gold chain, no planner call
PLANNED.update({f"{m}{CAP}{suffix}": PLANNED[m] for m in ("deepseek_plan", "small_plan")
                for suffix in ("", "_hidden")})
GUIDED = {"guided_deepseek_plan": ("deepseek", "qwen"), "guided_small_plan": ("qwen", "qwen"),
          "guided_gold": ("gold", "qwen"),
          "guided_deepseek_plan_deepseek": ("deepseek", "deepseek"),
          "guided_gold_deepseek": ("gold", "deepseek")}
GUIDED.update({"guided_deepseek_rev": ("deepseek_rev", "qwen"),
               "guided_deepseek_rev_deepseek": ("deepseek_rev", "deepseek")})
CHECKED = "_checked"  # suffix: the plan is offered as possibly wrong (GUIDED_CHECKED)
SETS = "_sets"  # suffix: #k is described as possibly several people (GUIDED_SETS)
COMMIT = "_commit"  # suffix: guidance with the no-abstain instruction (prompts.COMMIT)
GUIDED.update({f"{m}{suffix}": GUIDED[m] for m in ("guided_deepseek_plan",
                                                    "guided_small_plan",
                                                    "guided_deepseek_plan_deepseek",
                                                    "guided_gold", "guided_gold_deepseek")
               for suffix in (CHECKED, SETS, COMMIT)})
COMMITTED = ("small_commit", "deepseek_commit")
# Chains whose steps and final call carry the no-abstain instruction, and the gold plan
# answered by DeepSeek (retrieval-vs-given-context study).
PLANNED["gold_plan_deepseek"] = ("gold", "deepseek")
PLANNED.update({f"{m}{COMMIT}": PLANNED[m] for m in ("deepseek_plan", "small_plan", "gold_plan",
                                                    "deepseek_plan_deepseek",
                                                    "gold_plan_deepseek")})
DIRECTS = ("small", "deepseek") + COMMITTED
# Evidence contexts, as a "__<context>" suffix on a method (default: the row's passages):
#   ragq       direct or guided: the question's top-k retrieved passages
#   ragq15     direct: the question's top 15 (the budget of the per-sub-question union)
#   ragsub     guided: the union of every plan sub-question's top-k (#k left unfilled)
#   rag        chain: each step retrieves for its own sub-question, #k filled in
#   rag_final  chain: the rag steps (reused), then one final call over the question,
#              every step's answer and the union of their passages
CONTEXTS = {"ragq": "direct guided", "ragq15": "direct", "ragsub": "guided",
            "rag": "chain", "rag_final": "chain"}
RAG_K = 5
CHAINS = tuple(f"{m}{c}" for m in ("deepseek_plan", "small_plan", "gold_plan",
                                    "deepseek_plan_deepseek", "gold_plan_deepseek")
               for c in ("", COMMIT))  # the chains a context applies to
METHODS = DIRECTS + tuple(PLANNED) + tuple(GUIDED)
METHODS += tuple(f"{m}__{ctx}" for ctx, kinds in CONTEXTS.items()
                 for m in (DIRECTS if "direct" in kinds else ())
                 + (tuple(GUIDED) if "guided" in kinds else ())
                 + (CHAINS if "chain" in kinds else ()))
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


def answer_directly(row, who, log, rep, commit=False, ctx=None):
    pids = None
    if ctx in ("ragq", "ragq15"):
        row, pids = retrieved(row, [row["question"]], 15 if ctx == "ragq15" else RAG_K)
    call = log.call(model(who), f"{row['id']}/rep{rep}/direct",
                    prompts.direct_prompt(row, commit), DIRECT_CAP)
    answer, valid = prompts.final_answer(call["text"])
    if commit or ctx:  # a new cell: read leniently, as the other new cells are
        lenient = prompts.read_answer(call["text"])
        answer = lenient if lenient is not None else call["text"].strip()
    return answer, {"valid_format": valid, "finish_reason": call["finish_reason"],
                    "cost": cost([call]), **({"retrieved": pids} if pids is not None else {})}


def split_method(method):
    """-> (original method, mode, aggregation reader or None), e.g.
    "small_plan_context_dsread" -> ("small_plan", "context", "deepseek").
    The step cap stays in the original method: "deepseek_plan_cap4096_hidden" ->
    ("deepseek_plan_cap4096", "hidden", None)."""
    reader = None
    if method.endswith(DSREAD):
        method, reader = method[:-len(DSREAD)], "deepseek"
    for mode in MODES:
        if method.endswith("_" + mode):
            return method[:-len(mode) - 1], mode, reader
    return method, "chain", reader


def reuse(step_log, key, messages):
    """A saved step call, checked to have been asked with exactly these messages."""
    assert key in step_log.done, f"no saved step {key} in {step_log.path}"
    call = step_log.done[key]
    assert call["prompt_sha256"] == hashlib.sha256(json.dumps(
        messages, sort_keys=True).encode()).hexdigest(), f"step prompt differs: {key}"
    return call


def split_context(method):
    """-> (method without its context, context or None): "small__ragq" -> ("small", "ragq")."""
    base, sep, ctx = method.partition("__")
    if not sep:
        return method, None
    assert ctx in CONTEXTS, f"unknown context {ctx!r}"
    return base, ctx


def pool_of(row):
    return "phantomwiki" if "gold_chain" in row else "musique"


def query_text(text, row):
    """A retrieval query: the question or step without the dataset's answer note."""
    return text.removesuffix(row.get("answer_note", "")).strip()


def retrieved(row, queries, k=RAG_K):
    """(row with its passages replaced by the union of each query's top k, pool ids in
    order of first retrieval). The plan prompt never sees this row: only the answering
    calls do."""
    pids = []
    for hits in retrieval.retrieve(pool_of(row), [query_text(q, row) for q in queries], k):
        pids += [int(p) for p in hits if int(p) not in pids]
    return {**row, "paragraphs": retrieval.passages(pool_of(row), pids)}, pids


def gold_plan(row):
    """The gold chain as a plan: PhantomWiki's from its Prolog goals, MuSiQue's its
    own decomposition questions. The hop answers are never part of it."""
    if "gold_chain" in row:
        return phantomwiki.gold_plan(row)
    hops = row["question_decomposition"]
    steps = [hop["question"] for hop in hops]
    for i, step in enumerate(steps):  # a step may name only its own subject, never an answer
        assert not any(h["answer"].lower() in step.lower() for h in hops[i:]), (row["id"], step)
    return steps


def saved_plan(row, planner, plan_log, reuse_only):
    """-> (plan call, steps or None). reuse_only: the plan must be in the log, asked with
    exactly today's prompt; the planner is never loaded."""
    key = f"{row['id']}/plan"
    if reuse_only or (SMALL and (planner == "deepseek" or SMALL in SHARES_QWEN_PLANS)):
        assert key in plan_log.done, f"no saved {planner} plan for {key}"
        assert plan_log.done[key]["prompt_sha256"] == hashlib.sha256(json.dumps(
            prompts.plan_prompt(row), sort_keys=True).encode()).hexdigest(), key
        plan_call = plan_log.done[key]
    else:
        plan_call = plan_log.call(model(planner), key, prompts.plan_prompt(row), PLAN_CAP)
    return plan_call, prompts.parse_plan(plan_call["text"],
                                         row.get("max_steps", prompts.MAX_STEPS))


def revised_plan(row, plan_log):
    """DeepSeek's saved plan, checked against the question and corrected by DeepSeek
    (one call per question, logged in plan_log). -> (calls, steps or None). A revision
    that does not parse keeps the original plan; the record says so."""
    original_log = CallLog(plan_log.path.parent / "plans_deepseek.calls.jsonl")
    original, steps = saved_plan(row, "deepseek", original_log, reuse_only=True)
    if steps is None:
        return [original], None
    call = plan_log.call(model("deepseek"), f"{row['id']}/plan",
                         prompts.revise_prompt(row, steps), PLAN_CAP)
    revised = prompts.parse_plan(call["text"], row.get("max_steps", prompts.MAX_STEPS))
    return [original, call], revised or steps


def answer_guided(row, planner, answerer, plan_log, log, rep, variant="plain", ctx=None):
    """One call that answers the original question with a plan in context. ctx "ragq":
    the question's retrieved passages; "ragsub": the union of every plan sub-question's
    (#k left unfilled). The plan itself is the saved one, made from the row's passages."""
    if planner == "gold":
        steps, calls = gold_plan(row), []
    elif planner == "deepseek_rev":
        calls, steps = revised_plan(row, plan_log)
        if steps is None:
            return "", {"plan": None, "cost": cost(calls)}
    else:
        plan_call, steps = saved_plan(row, planner, plan_log, reuse_only=True)
        calls = [plan_call]
        if steps is None:  # unusable plan: the question counts as wrong, as in the chain
            return "", {"plan": None, "raw_plan": plan_call["text"], "cost": cost(calls)}
    evidence, pids = row, None
    if ctx == "ragq":
        evidence, pids = retrieved(row, [row["question"]])
    elif ctx == "ragsub":
        evidence, pids = retrieved(row, steps)
    call = log.call(model(answerer), f"{row['id']}/rep{rep}/guided",
                    prompts.guided_prompt(evidence, steps, variant), DIRECT_CAP)
    calls.append(call)
    answer = prompts.read_answer(call["text"])
    return answer if answer is not None else call["text"].strip(), {
        "plan": steps, "valid_format": prompts.final_answer(call["text"])[1],
        "finish_reason": call["finish_reason"], "output_tokens": call["output_tokens"],
        "cost": cost(calls), "plan_cost": cost(calls[:-1]),
        **({"retrieved": pids} if pids is not None else {})}


def answer_with_plan(row, planner, answerer, plan_log, log, rep, mode="chain", step_log=None,
                     reader=None, hop_cap=HOP_CAP, ctx=None, commit=False):
    """ctx "rag": each step answers over its own sub-question's retrieved passages (#k
    filled in); "rag_final": those steps, reused from step_log, then one final call over
    the question, every step's answer and the union of their passages. commit: steps and
    final call carry the no-abstain instruction."""
    if planner == "gold":
        plan_call, steps = None, gold_plan(row)
    else:
        # Only the original chain methods may draw a plan; every other one reuses it.
        plan_call, steps = saved_plan(row, planner, plan_log,
                                      reuse_only=mode != "chain" or hop_cap != HOP_CAP
                                      or ctx is not None or commit)
    calls = [plan_call] if plan_call else []
    if steps is None:  # unusable plan: the question counts as wrong
        return "", {"plan": None, "raw_plan": plan_call["text"], "cost": cost(calls)}
    if mode != "chain":
        return answer_and_aggregate(row, steps, answerer, plan_call, log, rep, mode, step_log,
                                    reader or answerer)
    note = row.get("answer_note", "")
    answers, hops, history, union = [], [], [], []
    for i, step in enumerate(steps):
        question = prompts.substitute(step, answers) + note
        key = f"{row['id']}/rep{rep}/hop{i}"
        evidence, pids = (retrieved(row, [question]) if ctx in ("rag", "rag_final")
                          else (row, None))
        messages = prompts.hop_prompt(question, evidence, commit)
        call = (reuse(step_log, key, messages) if ctx == "rag_final"
                else log.call(model(answerer), key, messages, hop_cap))
        answer, valid = prompts.final_answer(call["text"])
        if ctx or commit:  # a new cell: read leniently, as the other new cells are
            lenient = prompts.read_answer(call["text"])
            answer = lenient if lenient is not None else call["text"].strip()
        answers.append(answer)
        hops.append({"question": question, "answer": answer, "valid_format": valid,
                     "finish_reason": call["finish_reason"],
                     "output_tokens": call["output_tokens"],
                     **({"retrieved": pids} if pids is not None else {})})
        history.append(prompts.step_record(question, call["text"], note))
        union += [p for p in pids or () if p not in union]
        calls.append(call)
    info = {"plan": steps, "hops": hops, "plan_cost": cost(calls[:1] if plan_call else [])}
    if ctx != "rag_final":
        return answers[-1], {**info, "cost": cost(calls)}
    final_row = {**row, "paragraphs": retrieval.passages(pool_of(row), union)}
    agg = log.call(model(reader or answerer), f"{row['id']}/rep{rep}/aggregate",
                   prompts.aggregate_prompt(final_row, history, commit), DIRECT_CAP)
    calls.append(agg)
    answer = prompts.read_answer(agg["text"])
    return answer if answer is not None else agg["text"].strip(), {
        **info, "retrieved": union, "cost": cost(calls),
        "aggregate": {"valid_format": prompts.final_answer(agg["text"])[1],
                      "finish_reason": agg["finish_reason"],
                      "output_tokens": agg["output_tokens"]}}


def answer_and_aggregate(row, steps, answerer, plan_call, log, rep, mode, step_log, reader):
    """The hidden and context modes: run (or reuse, from step_log) the steps, then one
    aggregation call by reader."""
    note = row.get("answer_note", "")
    answers, history, hops, calls = [], [], [], [plan_call]
    for i, step in enumerate(steps):
        key = f"{row['id']}/rep{rep}/hop{i}"
        if mode == "hidden":  # the original step, exactly as it was asked
            question = prompts.substitute(step, answers) + note
            call = reuse(step_log, key, prompts.hop_prompt(question, row))
            answers.append(prompts.final_answer(call["text"])[0])
        else:
            question = prompts.substitute(step, answers) + note
            messages = prompts.context_hop_prompt(question, history, row)
            call = (reuse(step_log, key, messages) if step_log
                    else log.call(model(answerer), key, messages, HOP_CAP))
            answers.append(prompts.read_answer(call["text"]) or "UNKNOWN")
        record = prompts.step_record(question, call["text"], note)
        history.append(record)
        hops.append({"question": question, "answer": record["answer"],
                     "valid_format": prompts.final_answer(call["text"])[1],
                     "finish_reason": call["finish_reason"],
                     "output_tokens": call["output_tokens"]})
        calls.append(call)
    agg = log.call(model(reader), f"{row['id']}/rep{rep}/aggregate",
                   prompts.aggregate_prompt(row, history), DIRECT_CAP)
    calls.append(agg)
    answer = prompts.read_answer(agg["text"])
    return answer if answer is not None else agg["text"].strip(), {
        "plan": steps, "hops": hops,
        "aggregate": {"valid_format": prompts.final_answer(agg["text"])[1],
                      "finish_reason": agg["finish_reason"],
                      "output_tokens": agg["output_tokens"]},
        "cost": cost(calls), "plan_cost": cost([plan_call])}


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
    full = method
    method, ctx = split_context(full)
    planner, answerer = PLANNED.get(method) or GUIDED.get(method) or (None, None)
    base, mode, reader = split_method(method)
    if ctx:
        kind = ("direct" if method in DIRECTS else "guided" if method in GUIDED
                else "chain" if method in CHAINS else None)
        assert kind in CONTEXTS[ctx].split(), f"context {ctx} does not apply to {method}"
    pred, logs = PRED, LOGS
    if SMALL:
        assert "qwen" in (planner, answerer) or method.startswith("small"), \
            f"{method} uses no small model; reuse the saved run"
        pred, logs = PRED / SMALL, LOGS / SMALL
    log = CallLog(logs / f"{full}_rep{rep}.calls.jsonl")
    # Steps reused rather than run: hidden reuses the original method's steps; a
    # DeepSeek-read variant reuses the steps of the run it re-reads; rag_final reuses
    # the rag chain's steps.
    step_source = (f"{method}__rag" if ctx == "rag_final" else base if mode == "hidden"
                   else f"{base}_context" if reader else None)
    step_log = CallLog(logs / f"{step_source}_rep{rep}.calls.jsonl") if step_source else None
    # DeepSeek's plans are shared by every small model; a small model's own plans are not,
    # except for the FP8 build of Qwen3-4B, which answers the BF16 model's plans.
    shared = planner in ("deepseek", "deepseek_rev") or SMALL in SHARES_QWEN_PLANS
    plan_log = CallLog((LOGS if shared else logs)
                       / f"plans_{planner}.calls.jsonl") if planner not in (None, "gold") else None
    pred.mkdir(parents=True, exist_ok=True)
    predictions, details = [], []
    for n, row in enumerate(rows, 1):
        if method in GUIDED:
            variant = ("checked" if method.endswith(CHECKED) else "sets"
                       if method.endswith(SETS) else "commit" if method.endswith(COMMIT)
                       else "plain")
            answer, info = answer_guided(row, planner, answerer, plan_log, log, rep, variant,
                                         ctx)
        elif planner:
            answer, info = answer_with_plan(row, planner, answerer, plan_log, log, rep,
                                            mode, step_log, reader,
                                            DIRECT_CAP if base.endswith(CAP) else HOP_CAP,
                                            ctx, commit=method.endswith(COMMIT))
        else:
            answer, info = answer_directly(
                row, "qwen" if method.startswith("small") else "deepseek", log, rep,
                commit=method in COMMITTED, ctx=ctx)
        predictions.append({"id": row["id"], "predicted_answer": answer,
                            "predicted_support_idxs": [], "predicted_answerable": True})
        details.append({"id": row["id"], "method": full, "rep": rep,
                        "answer": answer, "gold": row["answer"], **info})
        print(f"{full} rep{rep} {n}/{len(rows)} {row['id']}: {answer!r} (gold {row['answer']!r})")
    suffix = ("" if limit is None else f".limit{limit}") + (
        f".only{'-'.join(map(str, only))}" if only else "")
    write_jsonl(pred / f"{full}_rep{rep}{suffix}.jsonl", predictions)
    write_jsonl(pred / f"{full}_rep{rep}{suffix}.details.jsonl", details)


def draw_plans(planner="deepseek"):
    """Write (or replay) the planner's plan for every question of the dataset, so that
    the methods that only reuse saved plans can run on a new sample."""
    plan_log = CallLog(LOGS / f"plans_{planner}.calls.jsonl")
    rows = [json.loads(line) for line in GOLD.open()]
    usable = 0
    for row in rows:
        _, steps = saved_plan(row, planner, plan_log, reuse_only=False)
        usable += steps is not None
    print(f"{usable}/{len(rows)} usable {planner} plans in {plan_log.path}")


def write_jsonl(path, rows):
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate():
    """Score every complete prediction file with the official evaluator.

    PhantomWiki answers are sets, which MuSiQue's evaluator cannot score, so
    they are scored with PhantomWiki's own rule (phantomwiki.score)."""
    if "phantomwiki" in GOLD.name:
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
    parser.add_argument("--draw-plans", action="store_true",
                        help="write DeepSeek's plan for every question of the dataset")
    parser.add_argument("--small", choices=sorted(SMALL_MODELS),
                        help="another small model on the local vLLM server instead of Qwen3-4B")
    parser.add_argument("--dataset", choices=DATASETS, help="another dataset instead of MuSiQue")
    args = parser.parse_args()
    if args.dataset:
        use_dataset(args.dataset)
    global SMALL, PRED
    SMALL = args.small
    if args.draw_plans:
        draw_plans()
    elif args.evaluate:
        if SMALL:
            PRED = PRED / SMALL
        evaluate()
    else:
        run(args.method, args.rep, args.limit, args.only)


if __name__ == "__main__":
    main()
