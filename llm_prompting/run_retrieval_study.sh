#!/usr/bin/env bash
# Every cell of the retrieval-vs-given-context study for one dataset and repetition.
#
#   llm_prompting/run_retrieval_study.sh <dataset> <rep> [lane ...]
#     dataset: dev48 (the MuSiQue pilot), dev300, phantomwiki48, phantomwiki146
#     lanes:   qwen-direct qwen-guided-ds qwen-guided-gold qwen-chain-ds qwen-chain-gold
#              ds-direct ds-guided ds-chain (default: all, in parallel), or "core": the
#              no-abstain cells the core comparison needs (9 Qwen, 6 DeepSeek)
#
# Run from external/musique with, for the Qwen lanes, the FP8 Qwen3-4B vLLM server on
# :8000. PYTHON must have transformers >= 4.51 (Qwen3 embeddings): on Hephaestus the cope
# venv (the default), on the laptop PYTHON=../../.venv-vllm/bin/python. Cells inside a lane run in order (a __rag chain before
# its __rag_final); lanes run in parallel. A cell already in the logs is replayed, not
# re-run. Each lane prints "done <cell>" or "FAILED <cell>"; errors go to
# logs/cs470/retrieval_study_<dataset>_rep<rep>_<lane>.err.
set -u
dataset=$1 rep=$2; shift 2
PYTHON=${PYTHON:-../cope/venv/bin/python}
flag=""; [ "$dataset" != dev48 ] && flag="--dataset $dataset"
with_commit() {  # <base> [contexts...]: the base and its no-abstain twin, each per context
  local base=$1; shift
  for m in "$base" "${base}_commit"; do
    for c in "" "$@"; do echo "$m${c:+__$c}"; done
  done
}
declare -A LANES
LANES[qwen-direct]="$(with_commit small ragq ragq15)"
LANES[qwen-guided-ds]="$(with_commit guided_deepseek_plan ragq ragsub)"
LANES[qwen-guided-gold]="$(with_commit guided_gold ragq ragsub)"
LANES[qwen-chain-ds]="deepseek_plan deepseek_plan_hidden $(with_commit deepseek_plan rag rag_final | grep -vx deepseek_plan)"
LANES[qwen-chain-gold]="$(with_commit gold_plan rag rag_final)"
LANES[ds-direct]="$(with_commit deepseek ragq ragq15)"
LANES[ds-guided]="$(with_commit guided_deepseek_plan_deepseek ragq ragsub) $(with_commit guided_gold_deepseek ragq ragsub)"
LANES[ds-chain]="deepseek_plan_deepseek deepseek_plan_deepseek_hidden $(with_commit deepseek_plan_deepseek rag rag_final | grep -vx deepseek_plan_deepseek) $(with_commit gold_plan_deepseek rag rag_final)"
# The core set: no-abstain only; direct (given, retrieved 15), DeepSeek's plan in one call
# with per-sub-question retrieval, DeepSeek's plan as steps (given, retrieving, retrieving
# + final call), and for Qwen the gold plan's one call and retrieving steps (upper bound).
LANES[qwen-core-direct]="small_commit small_commit__ragq15"
LANES[qwen-core-guided]="guided_deepseek_plan_commit__ragsub guided_gold_commit__ragsub"
LANES[qwen-core-chain-ds]="deepseek_plan_commit deepseek_plan_commit__rag deepseek_plan_commit__rag_final"
LANES[qwen-core-chain-gold]="gold_plan_commit__rag gold_plan_commit__rag_final"
LANES[ds-core]="deepseek_commit deepseek_commit__ragq15 guided_deepseek_plan_deepseek_commit__ragsub deepseek_plan_deepseek_commit deepseek_plan_deepseek_commit__rag deepseek_plan_deepseek_commit__rag_final"
lanes=("$@")
if [ ${#lanes[@]} -eq 0 ]; then
  lanes=(qwen-direct qwen-guided-ds qwen-guided-gold qwen-chain-ds qwen-chain-gold ds-direct
         ds-guided ds-chain)
elif [ "${lanes[0]}" = core ]; then
  lanes=(qwen-core-direct qwen-core-guided qwen-core-chain-ds qwen-core-chain-gold ds-core)
fi
run_lane() {
  local lane=$1 small=""
  [[ $lane == qwen-* ]] && small="--small qwen3-4b-fp8"
  local err="logs/cs470/retrieval_study_${dataset}_rep${rep}_${lane}.err"
  for cell in ${LANES[$lane]}; do
    if "$PYTHON" -m llm_prompting.run $small $flag --method "$cell" --rep "$rep" \
        > /dev/null 2>> "$err"; then echo "done $lane $cell"; else echo "FAILED $lane $cell"; fi
  done
  echo "LANE DONE $lane"
}
for lane in "${lanes[@]}"; do run_lane "$lane" & done
wait
echo "ALL LANES DONE $dataset rep$rep"
