"""Atomic hops: does Qwen execute DeepSeek's PhantomWiki plans when every step is a
single lookup?

The failure analysis found that, on DeepSeek's correct plans, Qwen3-4B fails mostly
at steps that ask about several people at once (pasting an answer set into the next
step) and at steps that run out of their 640-token budget after finding the answer,
whose unfinished text is then passed on. This runs DeepSeek's saved plans again under
three conditions, all with the same local engine, so they differ only in the step
format:

  baseline      the original step: the whole answer set pasted in, reason first and
                FINAL_ANSWER last, unfinished text passed on (as in run.py)
  answer_first  the same calls, but FINAL_ANSWER comes first; a step with no answer
                is retried once, then becomes UNKNOWN instead of passing text on
  atomic        answer_first, plus one call per person: a step about n people is n
                single lookups whose answers are merged in code
  atomic_reason one call per person as in atomic, but the original reason-first step
                prompt; retry and UNKNOWN as in answer_first. Separates the effect of
                splitting by person from the effect of answering first.

Plans are DeepSeek's plans from the original run (logs/cs470/phantomwiki48/
plans_deepseek.calls.jsonl), reused unchanged; no hosted model is called. All 48
questions are run (3 repetitions); the analysis restricts to the plans that are
correct.

Engine: Qwen3-4B-Instruct-2507 in Qwen's released FP8 build on vLLM (the original
runs used BF16 transformers on a 16 GB card; this laptop has 8 GB), FP8 KV cache,
temperature 0.4, top_p 0.9, 640 output tokens per step, as in run.py. Because the
engine differs, compare the conditions with each other, not with the original run.

Outputs (resume-safe: a logged call is never generated again):
  logs/cs470/phantomwiki48/atomic/<condition>.calls.jsonl     every call
  predictions/cs470/phantomwiki48/atomic/<condition>_rep<N>.details.jsonl
  predictions/cs470/phantomwiki48/atomic/<condition>_rep<N>.jsonl (+ .metrics.json)

Run from this directory, inside the workspace's cope shell, with .venv-vllm:
    nix develop ../..#cope -c ../../.venv-vllm/bin/python -m llm_prompting.atomic [--limit 2]
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from llm_prompting import prompts
from llm_prompting.phantomwiki import score

GOLD = Path("data/cs470_phantomwiki48.jsonl")
PLANS = Path("logs/cs470/phantomwiki48/plans_deepseek.calls.jsonl")
LOGS = Path("logs/cs470/phantomwiki48/atomic")
PRED = Path("predictions/cs470/phantomwiki48/atomic")
CONDITIONS = ("baseline", "answer_first", "atomic", "atomic_reason")
REPS = (0, 1, 2)

MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
REVISION = "8591804019c8b22094c3b5b4454e0edc05dffc98"
TEMPERATURE, TOP_P, HOP_CAP = 0.4, 0.9, 640
MAX_MODEL_LEN = 10240  # 51 articles (~8.3k tokens) + a step + 640 output tokens

ANSWER_FIRST = (
    "Start your reply with a single line FINAL_ANSWER: followed by only the shortest "
    "answer phrase, with no explanation on that line. If the supplied evidence cannot "
    "support an answer, put UNKNOWN on that line. After that line, add at most three "
    "short sentences naming the passages you used."
)


# ------------------------------------------------------------------ engine
class Engine:
    """vLLM Qwen3-4B with a resume-safe call log per condition."""

    def __init__(self):
        import os

        # top_p < 1 would select FlashInfer's sampler, which JIT-compiles with nvcc
        # (absent here); vLLM's PyTorch sampler applies the same top-p.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM

        # Triton attention supports the FP8 KV cache without FlashInfer, whose kernels
        # JIT-compile with nvcc (absent on this NixOS machine).
        self.llm = LLM(model=MODEL, revision=REVISION, dtype="bfloat16", kv_cache_dtype="fp8",
                       attention_backend="TRITON_ATTN", seed=0,
                       max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90,
                       max_num_seqs=64, max_num_batched_tokens=4096,
                       enable_prefix_caching=True)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, requests):
        """requests: [(key, messages)] -> [record] in the same order."""
        from vllm import SamplingParams

        texts = [self.tokenizer.apply_chat_template(m, tokenize=False,
                                                    add_generation_prompt=True)
                 for _, m in requests]
        params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=HOP_CAP,
                                 seed=seed_of(k)) for k, _ in requests]
        start = time.monotonic()
        outs = self.llm.generate(texts, params, use_tqdm=False)
        seconds = round((time.monotonic() - start) / max(len(requests), 1), 3)
        return [{
            "key": k, "model": MODEL, "text": o.outputs[0].text,
            "input_tokens": len(o.prompt_token_ids),
            "output_tokens": len(o.outputs[0].token_ids),
            "finish_reason": o.outputs[0].finish_reason,
            "seconds": seconds,  # batch wall time per call
            "seed": seed_of(k),
            "prompt_sha256": hashlib.sha256(
                json.dumps(m, sort_keys=True).encode()).hexdigest(),
            "question": json.loads(m[1]["content"].split("\n")[0])["question"],
        } for (k, m), o in zip(requests, outs)]


class CallLog:
    def __init__(self, path, engine):
        self.path, self.engine = Path(path), engine
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {}
        if self.path.exists():
            for line in self.path.open():
                record = json.loads(line)
                self.done[record["key"]] = record

    def run(self, requests):
        todo = [(k, m) for k, m in requests if k not in self.done]
        if todo:
            for record in self.engine().generate(todo):
                with self.path.open("a") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.done[record["key"]] = record
        return {k: self.done[k] for k, _ in requests}


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------- prompts
def answer_first_prompt(question, row):
    return prompts.messages(prompts.evidence(question, row) + "\n" + ANSWER_FIRST)


def first_answer(text):
    """The first FINAL_ANSWER line (answer-first format); None if there is none."""
    m = re.search(r"(?im)^[ \t]*FINAL[_ ]ANSWER:[ \t]*([^\r\n]*)", text)
    return m.group(1).strip() if m and m.group(1).strip() else None


def last_answer(text):
    """The last FINAL_ANSWER line (reason-first format); None if there is none."""
    found = re.findall(r"(?im)^[ \t]*FINAL[_ ]ANSWER:[ \t]*([^\r\n]*)", text)
    return found[-1].strip() if found and found[-1].strip() else None


def parser(cond):
    return last_answer if cond == "atomic_reason" else first_answer


def split_set(answer):
    """Answer string -> list of distinct items, UNKNOWN dropped, order kept."""
    items, seen = [], set()
    for a in answer.split(","):
        a = a.strip().rstrip(".")
        if a and a.upper() != "UNKNOWN" and a.lower() not in seen:
            seen.add(a.lower())
            items.append(a)
    return items


# ------------------------------------------------------------------- steps
def step_requests(cond, chain, i, step):
    """The calls for step i of one chain: [(key, messages, person or None)]."""
    row, rep = chain["row"], chain["rep"]
    note = row.get("answer_note", "")
    base = f"{row['id']}/rep{rep}/{cond}/hop{i}"
    if cond == "baseline":
        q = prompts.substitute(step, chain["answers"]) + note
        return [(base, prompts.hop_prompt(q, row), None)]
    prompt = prompts.hop_prompt if cond == "atomic_reason" else answer_first_prompt
    refs = sorted({int(n) for n in re.findall(r"#(\d+)", step)})
    if cond == "answer_first" or not refs:
        q = prompts.substitute(step, chain["answers"]) + note
        return [(base, prompt(q, row), None)]
    # atomic: one call per person in the last referenced step's answer set
    k = refs[-1]
    people = chain["sets"][k - 1]
    out = []
    for j, person in enumerate(people):
        answers = list(chain["answers"])
        answers[k - 1] = person
        q = prompts.substitute(step, answers) + note
        out.append((f"{base}/p{j}", prompt(q, row), person))
    return out


def run_condition(cond, chains, engine):
    log = CallLog(LOGS / f"{cond}.calls.jsonl", engine)
    depth = max(len(c["plan"]) for c in chains)
    for i in range(depth):
        live = [c for c in chains if i < len(c["plan"])]
        reqs = {id(c): step_requests(cond, c, i, c["plan"][i]) for c in live}
        flat = [(k, m) for rs in reqs.values() for k, m, _ in rs]
        got = log.run(flat)
        if cond != "baseline":  # retry once when a call gave no answer
            parse = parser(cond)
            retry = [(k + "/retry", m) for k, m in flat if parse(got[k]["text"]) is None]
            got.update(log.run(retry))
        for c in live:
            calls = []
            for k, _, person in reqs[id(c)]:
                rec = got[k]
                if cond == "baseline":
                    answer, valid = prompts.final_answer(rec["text"])
                    retried = False
                else:
                    answer = parse(rec["text"])
                    retried = answer is None
                    if retried:
                        rec = got[k + "/retry"]
                        answer = parse(rec["text"])
                    valid = answer is not None
                    answer = answer or "UNKNOWN"
                calls.append({"key": rec["key"], "person": person, "question": rec["question"],
                              "answer": answer, "valid_format": valid, "retried": retried,
                              "finish_reason": rec["finish_reason"],
                              "output_tokens": rec["output_tokens"]})
            if cond == "baseline":
                answer = calls[0]["answer"]  # raw text passed on when invalid, as in run.py
                items = split_set(answer)
            else:
                items = []
                for call in calls:
                    for a in split_set(call["answer"]):
                        if a.lower() not in {x.lower() for x in items}:
                            items.append(a)
                answer = ", ".join(items) if items else "UNKNOWN"
            c["answers"].append(answer)
            c["sets"].append(items)
            c["hops"].append({"template": c["plan"][i], "calls": calls, "answer": answer})


def load_plans(rows):
    plans = {}
    for line in PLANS.open():
        record = json.loads(line)
        plans[record["key"].split("/")[0]] = record["text"]
    return {r["id"]: prompts.parse_plan(plans[r["id"]], r.get("max_steps", prompts.MAX_STEPS))
            for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="smoke test on the first N questions")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    rows = [json.loads(line) for line in GOLD.open()][:args.limit]
    plans = load_plans(rows)
    engine_box = []

    def engine():
        if not engine_box:
            engine_box.append(Engine())
        return engine_box[0]

    PRED.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit is None else f".limit{args.limit}"
    for cond in args.conditions.split(","):
        chains = [{"row": r, "rep": rep, "plan": plans[r["id"]], "answers": [], "sets": [],
                   "hops": []} for rep in REPS for r in rows if plans[r["id"]]]
        start = time.monotonic()
        run_condition(cond, chains, engine)
        for rep in REPS:
            mine = [c for c in chains if c["rep"] == rep]
            details = [{"id": c["row"]["id"], "condition": cond, "rep": rep,
                        "plan": c["plan"], "hops": c["hops"], "answer": c["answers"][-1],
                        "gold": c["row"]["answer_set"],
                        **score(c["answers"][-1], c["row"]["answer_set"])} for c in mine]
            write_jsonl(PRED / f"{cond}_rep{rep}{suffix}.details.jsonl", details)
            write_jsonl(PRED / f"{cond}_rep{rep}{suffix}.jsonl",
                        [{"id": d["id"], "predicted_answer": d["answer"],
                          "predicted_support_idxs": [], "predicted_answerable": True}
                         for d in details])
            n = len(details)
            (PRED / f"{cond}_rep{rep}{suffix}.metrics.json").write_text(json.dumps({
                "answer_em": sum(d["em"] for d in details) / n,
                "answer_f1": sum(d["f1"] for d in details) / n,
                "per_question": {d["id"]: {"em": d["em"], "f1": d["f1"]} for d in details},
            }, indent=1))
            print(f"{cond} rep{rep}: EM {sum(d['em'] for d in details) / n:.3f} "
                  f"over {n} questions ({time.monotonic() - start:.0f} s so far)", flush=True)


def write_jsonl(path, rows):
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
