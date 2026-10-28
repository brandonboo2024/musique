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
| `deepseek_plan_deepseek` | DeepSeek → DeepSeek, one hop at a time (the same plans as `deepseek_plan`) | 1 plan + 1 per hop |

- Every model call sees the same 20 paragraphs. The `is_supporting` label is removed first.
- Planned methods (`deepseek_plan_deepseek` reads the same plan log as `deepseek_plan`, so both answer identical plans):
  - The planner writes one plan per question, which is reused unchanged in all three repetitions.
  - Each hop's `#k` is replaced by the answerer's own answer to hop k.
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

## Aggregation modes (October 2026 follow-up)
In the planned methods, a step learns an earlier answer only through `#k`, and the last step's answer is final. Each planned method therefore also has two variants that end the chain with one aggregation call. That call sees the original question, the passages and every subquestion with its answer.

| Method | Steps | Final answer |
|---|---|---|
| `<m>_hidden` | exactly `<m>`'s steps (saved calls reused and checked by prompt hash) | aggregation call |
| `<m>_context` | each step also sees all earlier subquestions and their answers | aggregation call |
| `<m>_hidden_dsread`, `<m>_context_dsread` | the steps of `<m>` / `<m>_context`, reused | aggregation call by DeepSeek |

- `<m>` is `deepseek_plan`, `small_plan` or `deepseek_plan_deepseek`. The `_dsread` variants exist only where Qwen answers the steps.
- All variants reuse the saved plans and never draw a new one.
- They read answers with `prompts.read_answer`, which also accepts a `FINAL_ANSWER:` after other text on its line or in Markdown bold. A step with no answer line is shown to later calls as "NO ANSWER" rather than as its unfinished text.
- The original methods keep `final_answer`, so their saved runs reproduce unchanged.
- Offline checks, with no model calls: `python -m unittest llm_prompting.test_aggregate`.

## Outputs
- `predictions/cs470/<method>_rep<N>.jsonl`: official format.
- `predictions/cs470/<method>_rep<N>.metrics.json`: `evaluate_v1.0.py` output.
- `predictions/cs470/<method>_rep<N>.details.jsonl`: plan, hop questions and answers, each hop's finish reason and output tokens (`length` means the hop hit its cap and its unfinished text was passed on), tokens and US$ per question. Runs before 18 Oct 2026 lack the per-hop finish reason; it is in the call log.
- `logs/cs470/*.calls.jsonl`: every model call (output text, tokens, seconds, cost). Prompts are stored as a hash and rebuilt from the code and data.

## Scorer check
- Gold answers as predictions give answer EM/F1 1.0; blank answers give 0.0.
- MuSiQue's released model predictions (Google Drive) could not be downloaded via `gdown` on 2026-10-09, so the README numbers were not re-checked.

## PhantomWiki (second dataset)
`phantomwiki.py` builds `data/cs470_phantomwiki48.jsonl` from PhantomWiki v1
(Gong et al., 2025), instance `depth_20_size_50_seed_1` on Hugging Face
(`kilian-group/phantom-wiki-v1`): 8 questions at each difficulty 3, 6, 9, 12,
15, 18 (`sample_phantomwiki48.json`). All 51 articles are the passages of
every call. Each row also carries the gold answer set after every relation
(`gold_chain`), computed with SWI-Prolog from the instance's own facts; it is
used only by `trace.py` and never reaches a model.

Differences from MuSiQue: answers are comma-separated sets (every question
and hop ends with an instruction to list them all), plans may have up to 12
steps, and scoring is PhantomWiki's rule (set EM and set F1, as in its
`phantom_eval/score.py`), not `evaluate_v1.0.py`.

```sh
# raw files in ../../data/raw/hop_candidates/: pw_qa_50, pw_corpus_50, pw_db_50 (.parquet)
nix develop ../..#cope -c nix shell nixpkgs#swi-prolog -c ../cope/venv/bin/python -m llm_prompting.phantomwiki ../../data/raw/hop_candidates
python -m llm_prompting.run --dataset phantomwiki48 --method small --rep 0
python -m llm_prompting.run --dataset phantomwiki48 --evaluate
```
Outputs go to `predictions/cs470/phantomwiki48/` and `logs/cs470/phantomwiki48/`.

`pilot_data.py` holds the 8-question pilots (PhantomWiki, MoreHopQA,
MultiHop-RAG) used to choose the dataset (`--dataset phantomwiki|morehopqa|multihoprag`).

## Tracing failures
`python -m llm_prompting.trace [--dataset phantomwiki48] [--rep N]` writes
`predictions/cs470/[<dataset>/]trace_repN.org`: per question, the gold chain,
then for each method the plan, each hop's exact input (passages elided) and
raw output, the answer read from it, and which gold step that answer equals.
The first hop that matches no gold step is usually where the chain broke.

## Small runs by hand
Pick question numbers from a trace's summary table (Q1-Q48), and use a
repetition number above 2 so that new samples are drawn (calls already in a
log are replayed, not re-run) and RESULTS.org (reps 0-2) is left untouched:
```sh
python -m llm_prompting.run --dataset phantomwiki48 --method deepseek --rep 5 --only 9,17,41
python -m llm_prompting.trace --dataset phantomwiki48 --rep 5 --suffix .only9-17-41
```
Plans are logged per question, not per repetition, so a new repetition
reuses the same plans.

## Atomic hops (branch `cs470-atomic-hops`)
`atomic.py` re-executes DeepSeek's saved PhantomWiki plans with Qwen3-4B under four step
formats (`baseline`, `answer_first`, `atomic`, `atomic_reason`; see its docstring), to test
whether one lookup per call fixes Qwen's execution of correct plans. It runs locally on
vLLM with Qwen's FP8 build (an 8 GB GPU cannot hold the BF16 model), so compare the
conditions with each other rather than with the runs above.

```sh
# data/cs470_phantomwiki48.jsonl: rebuilt with phantomwiki.py (see above)
nix develop ../..#cope -c ../../.venv-vllm/bin/python -m llm_prompting.atomic
```
Outputs: `logs/cs470/phantomwiki48/atomic/<condition>.calls.jsonl` (every call, with the
step question) and `predictions/cs470/phantomwiki48/atomic/<condition>_rep<N>.*`. Step
scoring against the PhantomWiki oracle is in the CS470 workspace
(`analysis/failure_cases_lib.py`, `atomic_hop_table`); results are in its RESULTS.org.

## Planning as guidance (`guided_*`, `gold_plan`, `*_cap4096`)
These cells separate the plan from the splitting into steps, and from the step budget
(see `run.py`'s docstring for each method). A guided method answers the original
question in one call with a saved plan (or the gold chain written as a plan) in context.
`gold_plan` runs the gold plan as a chain. `deepseek_plan_cap4096` and
`small_plan_cap4096` rerun the chains with the direct cap per step, and `_hidden` re-reads
them with the final call. No plan is drawn again.

The Qwen cells ran on the 8 GB laptop with `--small qwen3-4b-fp8` (Qwen3-4B-Instruct-2507,
FP8 build), which answers the BF16 model's saved Qwen plans. The Qwen baselines (`small`,
the chains and `_hidden`) were rerun the same way, so all Qwen rows share one engine:
```sh
# from the workspace root
# on the RTX 5070 Ti (Blackwell) also VLLM_USE_DEEP_GEMM=0: DeepGEMM JIT-compiles its FP8
# kernels and needs a CUDA toolkit home, which NixOS does not provide
VLLM_USE_FLASHINFER_SAMPLER=0 nix develop .#cope -c .venv-vllm/bin/vllm serve \
  Qwen/Qwen3-4B-Instruct-2507-FP8 --revision 8591804019c8b22094c3b5b4454e0edc05dffc98 \
  --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --seed 0 --max-num-seqs 16
# then, from here
for d in "" "--dataset phantomwiki48"; do
  for m in small deepseek_plan small_plan deepseek_plan_hidden small_plan_hidden \
           guided_deepseek_plan guided_small_plan guided_gold gold_plan \
           deepseek_plan_cap4096 small_plan_cap4096 \
           deepseek_plan_cap4096_hidden small_plan_cap4096_hidden; do
    for r in 0 1 2; do
      python -m llm_prompting.run --small qwen3-4b-fp8 $d --method $m --rep $r
    done
  done
done
python -m unittest llm_prompting.test_guided   # offline checks, no model calls
```
`*_cap4096_hidden` reuses the steps of `*_cap4096`, so it runs after it.

## Retrieval vs given context (`__<context>` methods)
Any direct, guided or chain method can take its evidence from retrieval instead of the
row's passages, with a `__<context>` suffix (see `run.py` and `retrieval.py`):
- `__ragq` / `__ragq15`: the question's top 5 / top 15 passages (direct; guided `__ragq`);
- `__ragsub`: the union of every plan sub-question's top 5, `#k` unfilled (guided);
- `__rag`: each chain step retrieves for its own sub-question, `#k` filled in;
- `__rag_final`: the `__rag` steps (reused), then one final call over every step's answer
  and the union of their passages.

Chains without abstaining (`deepseek_plan_commit`, `gold_plan_commit`, ...) and the gold
plan with DeepSeek steps (`gold_plan_deepseek`) complete the grid. The pools are every
distinct MuSiQue dev paragraph and the 51 PhantomWiki articles; queries are embedded on the
CPU (Qwen3-Embedding-0.6B), paragraph vectors once on the GPU and cached in
`data/cache/retrieval/`. Samples: `python -m llm_prompting.sample dev300` and
`python -m llm_prompting.phantomwiki ../../data/raw/hop_candidates phantomwiki146` (needs
`swipl`). New samples need DeepSeek's plans first:
```sh
python -m llm_prompting.run --dataset dev300 --draw-plans
python -m llm_prompting.run --small qwen3-4b-fp8 --dataset dev300 --method deepseek_plan__rag
python -m llm_prompting.run --small qwen3-4b-fp8 --dataset dev300 --method deepseek_plan__rag_final
python -m unittest llm_prompting.test_rag   # offline checks, no model calls, no GPU
```
`llm_prompting/run_retrieval_study.sh <dataset> <rep>` runs every cell of the study in
eight parallel lanes (five Qwen lanes on the vLLM server, three DeepSeek lanes).
