"""Draw the fixed 48-question MuSiQue-Ans dev sample (16 per hop count).

Writes llm_prompting/sample_dev48.json (committed) and data/cs470_dev48.jsonl,
the matching subset of the official dev file, in the same order, as
evaluate_v1.0.py requires.

    python -m llm_prompting.sample
"""

import json
import random
from pathlib import Path

DEV = Path("data/musique_ans_v1.0_dev.jsonl")
SAMPLE = Path("llm_prompting/sample_dev48.json")
GOLD = Path("data/cs470_dev48.jsonl")
SEED = 20261009
PER_HOP = 16


def read_dev():
    return [json.loads(line) for line in DEV.open() if line.strip()]


def draw():
    rows = read_dev()
    rng = random.Random(SEED)
    ids = []
    for hops in (2, 3, 4):
        pool = sorted(r["id"] for r in rows if r["id"].startswith(f"{hops}hop"))
        ids += sorted(rng.sample(pool, PER_HOP))
    return ids


def write_gold(ids):
    by_id = {r["id"]: r for r in read_dev()}
    with GOLD.open("w") as out:
        for i in ids:
            out.write(json.dumps(by_id[i]) + "\n")


def main():
    if SAMPLE.exists():
        ids = json.loads(SAMPLE.read_text())["ids"]  # never redraw a committed sample
    else:
        ids = draw()
        SAMPLE.write_text(json.dumps(
            {"source": str(DEV), "seed": SEED, "per_hop": PER_HOP, "ids": ids}, indent=1))
    write_gold(ids)
    print(f"{len(ids)} questions -> {SAMPLE}, {GOLD}")


if __name__ == "__main__":
    main()
