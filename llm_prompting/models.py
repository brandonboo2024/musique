"""The two models and a resume-safe call log.

Every call is appended to a .calls.jsonl log under logs/cs470/. A call whose key
is already logged is never repeated, so an interrupted run resumes without
re-billing DeepSeek or re-sampling Qwen.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

QWEN = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"  # pinned snapshot
DEEPSEEK = "deepseek-flash"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
PRICE_IN, PRICE_OUT = 0.30, 1.20  # US$ per million tokens, DeepSeek Flash peak rates
TEMPERATURE, TOP_P = 0.4, 0.9


class CallLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {}
        if self.path.exists():
            with self.path.open() as lines:
                for line in lines:
                    record = json.loads(line)
                    self.done[record["key"]] = record

    def call(self, model, key, messages, max_tokens):
        if key in self.done:
            return self.done[key]
        record = {"key": key, **model.generate(key, messages, max_tokens)}
        record["prompt_sha256"] = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()).hexdigest()
        with self.path.open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.done[key] = record
        return record


class Qwen:
    """Qwen3-4B, local, Hugging Face transformers in BF16."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN, revision=QWEN_REVISION, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def generate(self, key, messages, max_tokens):
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to("cuda")
        ids = inputs["input_ids"]
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)  # same key -> same sample
        start = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=TEMPERATURE, top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, ids.shape[1]:]
        hit_eos = bool((new == self.tokenizer.eos_token_id).any()) or \
            new[-1].item() in self.tokenizer.all_special_ids
        return {
            "model": QWEN,
            "text": self.tokenizer.decode(new, skip_special_tokens=True),
            "input_tokens": ids.shape[1],
            "output_tokens": int(new.shape[0]),
            "finish_reason": "stop" if hit_eos else "length",
            "seconds": round(time.monotonic() - start, 3),
            "usd": 0.0,
            "seed": seed,
        }


# Other small models, served by a local vLLM OpenAI-compatible server (the 8 GB
# laptop GPU needs quantised builds: AWQ 4-bit for Qwen3-8B, FP8 for Gemma 3 4B and
# the original April Qwen3-4B; Qwen2.5-1.5B, COPE's small model, fits in BF16).
# name -> (served model, pinned revision).
SMALL_MODELS = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct",
                     "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen3-4b-orig": ("Qwen/Qwen3-4B-FP8", "96b30dc13593a244a5e59e84687309f53c375cfa"),
    "qwen3-8b": ("Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79"),
    "gemma3-4b": ("RedHatAI/gemma-3-4b-it-FP8-dynamic",
                  "e8535f618f3ff62cd745395b05eb8cbde0851b71"),
    # Qwen3-4B-Instruct-2507 itself, in Qwen's FP8 build (as in atomic.py): the 8 GB
    # stand-in for the BF16 engine. It answers Qwen3-4B's saved plans, never its own.
    "qwen3-4b-fp8": ("Qwen/Qwen3-4B-Instruct-2507-FP8",
                     "8591804019c8b22094c3b5b4454e0edc05dffc98"),
}
SHARES_QWEN_PLANS = {"qwen3-4b-fp8"}  # these reuse the BF16 Qwen3-4B's saved plans
SMALL_URL = os.getenv("SMALL_URL", "http://127.0.0.1:8000/v1/chat/completions")


class LocalSmall:
    """A SMALL_MODELS entry on the local vLLM server; same sampling and seeding as Qwen."""

    def __init__(self, name):
        self.name = name
        self.model, self.revision = SMALL_MODELS[name]

    def generate(self, key, messages, max_tokens):
        """One call. A prompt too long for the server's context with max_tokens more (a
        chain step whose subject is an earlier step's unfinished text) is retried with
        the output cap that still fits; one that does not fit at all is recorded with no
        output and finish_reason "context_length". The record keeps the cap used."""
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)  # as in Qwen
        start, cap = time.monotonic(), max_tokens
        for attempt in range(2):
            payload = {"model": self.model, "messages": messages, "max_tokens": cap,
                       "temperature": TEMPERATURE, "top_p": TOP_P, "seed": seed,
                       "chat_template_kwargs": {"enable_thinking": False}}
            request = urllib.request.Request(
                SMALL_URL, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    raw = json.load(response)
                break
            except urllib.error.HTTPError as error:
                message = error.read().decode(errors="replace")
                if error.code != 400 or "maximum context length" not in message or attempt:
                    raise
                # the exact prompt length, from the server's tokenizer
                count = json.loads(urllib.request.urlopen(urllib.request.Request(
                    SMALL_URL.replace("/v1/chat/completions", "/tokenize"),
                    data=json.dumps({"model": self.model, "messages": messages,
                                     "add_generation_prompt": True,
                                     "chat_template_kwargs": {"enable_thinking": False}
                                     }).encode(),
                    headers={"Content-Type": "application/json"}), timeout=600).read())
                cap = count["max_model_len"] - count["count"]
                if cap < 1:
                    return {"model": f"{self.model}@{self.revision}", "text": "",
                            "input_tokens": count["count"], "output_tokens": 0,
                            "finish_reason": "context_length",
                            "seconds": round(time.monotonic() - start, 3), "usd": 0.0,
                            "seed": seed, "max_tokens_used": 0}
        usage = raw["usage"]
        return {
            "model": f"{self.model}@{self.revision}",
            "text": raw["choices"][0]["message"]["content"] or "",
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "finish_reason": raw["choices"][0]["finish_reason"],
            "seconds": round(time.monotonic() - start, 3),
            "usd": 0.0,
            "seed": seed,
            **({"max_tokens_used": cap} if cap != max_tokens else {}),
        }


class DeepSeek:
    """DeepSeek Flash via the hosted API, thinking disabled."""

    def __init__(self):
        self.key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") \
            or os.getenv("OPEN_API_KEY")
        if not self.key:
            raise SystemExit("No DeepSeek API key: source the project's .env.local")

    def generate(self, key, messages, max_tokens):
        payload = {"model": DEEPSEEK, "messages": messages, "max_tokens": max_tokens,
                   "temperature": TEMPERATURE, "top_p": TOP_P,
                   "thinking": {"type": "disabled"}}
        request = urllib.request.Request(
            DEEPSEEK_URL, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=180) as response:  # no retry: may be billed
            raw = json.load(response)
        usage = raw["usage"]
        return {
            "model": raw["model"],
            "text": raw["choices"][0]["message"]["content"] or "",
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "finish_reason": raw["choices"][0]["finish_reason"],
            "seconds": round(time.monotonic() - start, 3),
            "usd": (usage["prompt_tokens"] * PRICE_IN
                    + usage["completion_tokens"] * PRICE_OUT) / 1e6,
        }
