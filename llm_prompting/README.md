# CS470: training-free LLM prompting on MuSiQue

This folder is the only addition to the upstream MuSiQue repository on the
`cs470` branch. It does **not** use MuSiQue's trained models. It prompts
off-the-shelf LLMs and scores them with MuSiQue's unmodified
`evaluate_v1.0.py`.

## What comes from MuSiQue
- `data/musique_ans_v1.0_dev.jsonl`: questions, 20 paragraphs each, gold answers and aliases (official v1.0 zip).
- The prediction format and `evaluate_v1.0.py` (answer EM and F1).

## Sample
`sample_dev48.json`: 48 MuSiQue-Ans **dev** questions, 16 each of 2, 3 and 4 hops, drawn with seed 20261009 by `sample.py`. `data/cs470_dev48.jsonl` is the matching subset of the official dev file, in the same order, which `evaluate_v1.0.py` requires.

## Methods
| Method | Who plans → who answers | Calls per question |
|---|---|---|
| `small` | — → Qwen3-4B-Instruct-2507 (local, BF16) | 1 |
| `deepseek` | — → DeepSeek Flash (`deepseek-flash`, thinking off) | 1 |
| `deepseek_plan` | DeepSeek → Qwen3-4B, one hop at a time | 1 plan + 1 per hop |
| `small_plan` | Qwen3-4B → Qwen3-4B, one hop at a time | 1 plan + 1 per hop |

- Every model call sees the same 20 paragraphs. The `is_supporting` label is removed first.
- Planned methods:
  - The planner writes one plan per question, which is reused unchanged in all three repetitions.
  - Each hop's `#k` is replaced by Qwen's own answer to hop k.
  - The final answer is the last hop's answer.
  - An unusable plan (bad JSON, more than 4 steps, or a forward reference) counts as a wrong answer.
- Sampling: temperature 0.4, top_p 0.9, 3 repetitions. Qwen's seed is derived from each call's key, so reruns reproduce.
- Output caps: 4096 tokens for a direct answer, 512 for a plan, 640 for a hop.

## Prompts (`prompts.py`)
Taken verbatim from the earlier CS470 runs:
- `SYSTEM` + `REASON`: the earlier "Direct" prompt, which also served as the earlier per-hop prompt.
- `PLAN`: the earlier "generated plan" prompt.

Answers are read from a final `FINAL_ANSWER:` line, and UNKNOWN is allowed.

## Differences from the earlier runs
- Dev split instead of train, and a fresh sample.
- No final "integration" call after the hops: the last hop's answer is the answer.
- The plan parser also accepts a plan wrapped in a ```json code fence.
- No support prediction: `predicted_support_idxs` is empty, so `support_f1` is 0 by construction and is not reported.

## Run
From this directory, with the CS470 workspace's `.venv-coordination` Python:
```sh
export HF_HOME=$PWD/../../data/raw/huggingface HF_HUB_OFFLINE=1
set -a; . ../../.env.local; set +a          # DeepSeek key
python -m llm_prompting.sample              # writes data/cs470_dev48.jsonl
python -m llm_prompting.run --method small --rep 0 --limit 2   # smoke test
python -m llm_prompting.run --method small --rep 0
python -m llm_prompting.run --evaluate      # official scores -> *.metrics.json
```

## Outputs
- `predictions/cs470/<method>_rep<N>.jsonl`: official format.
- `predictions/cs470/<method>_rep<N>.metrics.json`: `evaluate_v1.0.py` output.
- `predictions/cs470/<method>_rep<N>.details.jsonl`: plan, hop questions and answers, tokens and US$ per question.
- `logs/cs470/*.calls.jsonl`: every model call (output text, tokens, seconds, cost). Prompts are stored as a hash and rebuilt from the code and data.

## Scorer check
- Gold answers as predictions give answer EM/F1 1.0; blank answers give 0.0.
- MuSiQue's released model predictions (Google Drive) could not be downloaded via `gdown` on 2026-10-09, so the README numbers were not re-checked.
