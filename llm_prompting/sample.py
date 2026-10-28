"""Draw the fixed MuSiQue-Ans dev samples.

- dev48 (default): 16 per hop count, the sample of every earlier experiment.
- dev300: 100 per hop count, drawn with its own seed from the questions not in dev48,
  for the retrieval-vs-given-context study.

Each writes llm_prompting/sample_<name>.json (committed) and data/cs470_<name>.jsonl,
the matching subset of the official dev file, in the same order, as
evaluate_v1.0.py requires. A committed sample is never redrawn.

    python -m llm_prompting.sample [dev300]
"""

import json
import random
from pathlib import Path

DEV = Path("data/musique_ans_v1.0_dev.jsonl")
SAMPLES = {  # name -> (seed, per hop count, samples whose questions are excluded)
    "dev48": (20261009, 16, ()),
    "dev300": (20261028, 100, ("dev48",)),
}


def read_dev():
    return [json.loads(line) for line in DEV.open() if line.strip()]


def sample_path(name):
    return Path(f"llm_prompting/sample_{name}.json")


def draw(name):
    seed, per_hop, excluded = SAMPLES[name]
    taken = {i for other in excluded for i in json.loads(sample_path(other).read_text())["ids"]}
    rows = read_dev()
    rng = random.Random(seed)
    ids = []
    for hops in (2, 3, 4):
        pool = sorted(r["id"] for r in rows
                      if r["id"].startswith(f"{hops}hop") and r["id"] not in taken)
        ids += sorted(rng.sample(pool, per_hop))
    return ids


def write_gold(name, ids):
    by_id = {r["id"]: r for r in read_dev()}
    with Path(f"data/cs470_{name}.jsonl").open("w") as out:
        for i in ids:
            out.write(json.dumps(by_id[i]) + "\n")


def main(name="dev48"):
    sample = sample_path(name)
    if sample.exists():
        ids = json.loads(sample.read_text())["ids"]  # never redraw a committed sample
    else:
        seed, per_hop, excluded = SAMPLES[name]
        ids = draw(name)
        sample.write_text(json.dumps({"source": str(DEV), "seed": seed, "per_hop": per_hop,
                                      "excludes": list(excluded), "ids": ids}, indent=1))
    write_gold(name, ids)
    print(f"{len(ids)} questions -> {sample}, data/cs470_{name}.jsonl")


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
