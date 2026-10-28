"""Retrieval over a fixed paragraph pool, for the retrieval-vs-given-context study.

Pools:
  musique      every distinct paragraph (title + text) of the 2,417 MuSiQue-Ans dev
               questions, so each question's own 20 paragraphs compete with the
               distractors of all the others.
  phantomwiki  the 51 articles of the PhantomWiki instance.

Retrievers:
  dense  Qwen3-Embedding-0.6B, last-token pooling, L2-normalised, cosine similarity;
         queries carry the model's instruction prefix. Paragraph embeddings are cached
         in data/cache/retrieval/.
  bm25   Okapi BM25 (k1 = 1.5, b = 0.75) over lower-cased word tokens, as a check.

Supporting paragraphs (for recall): MuSiQue's is_supporting labels; for PhantomWiki,
the articles of the people a gold chain step reads from (see phantomwiki_support).

    nix develop ../..#cope -c ../../.venv-vllm/bin/python -m llm_prompting.retrieval
"""

import json
import math
import re
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

EMBEDDER = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
TASK = "Given a question, retrieve Wikipedia passages that answer the question"
CACHE = Path("data/cache/retrieval")
DEV = Path("data/musique_ans_v1.0_dev.jsonl")
PHANTOMWIKI = Path("data/cs470_phantomwiki48.jsonl")


def pool_key(title, text):
    return title.strip() + "\n" + text.strip()


@cache
def pool(dataset):
    """[{pid, title, paragraph_text}] for a dataset's pool, in a fixed order."""
    seen, out = {}, []
    if dataset == "musique":
        rows = (json.loads(line) for line in DEV.open())
        paragraphs = (p for r in rows for p in r["paragraphs"])
    elif dataset == "phantomwiki":
        paragraphs = json.loads(PHANTOMWIKI.open().readline())["paragraphs"]
    else:
        raise ValueError(dataset)
    for p in paragraphs:
        key = pool_key(p["title"], p["paragraph_text"])
        if key not in seen:
            seen[key] = len(out)
            out.append({"pid": len(out), "title": p["title"],
                        "paragraph_text": p["paragraph_text"]})
    return out


@cache
def pid_of(dataset):
    return {pool_key(p["title"], p["paragraph_text"]): p["pid"] for p in pool(dataset)}


def support(dataset, row):
    """The pool ids a question's answer needs."""
    if dataset == "musique":
        ids = pid_of(dataset)
        return {ids[pool_key(p["title"], p["paragraph_text"])]
                for p in row["paragraphs"] if p["is_supporting"]}
    return phantomwiki_support(row)


def phantomwiki_support(row):
    """Articles a gold chain reads: the anchor (a named person, or the person found by an
    attribute) and every person a later step looks up. The last step's own answers are
    not read, so they are not included."""
    by_title = {p["title"]: p["pid"] for p in pool("phantomwiki")}
    names = set()
    chain = row["gold_chain"]
    for goal in chain:
        names |= set(re.findall(r'"([^"]+)"', goal["goal"])) & set(by_title)
    for step in chain[:-1]:
        names |= set(step["answers"]) & set(by_title)
    if chain and re.match(r'\w+\(Y_\d+, "', chain[0]["goal"]):  # attribute anchor
        names |= set(chain[0]["answers"]) & set(by_title)
    return {by_title[n] for n in names}


# ------------------------------------------------------------------ dense
class Embedder:
    """Qwen3-Embedding: left padding, last-token pooling, normalised vectors."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        name, revision = EMBEDDER
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision,
                                                       padding_side="left")
        self.model = AutoModel.from_pretrained(name, revision=revision,
                                               torch_dtype=torch.float16).cuda().eval()

    def __call__(self, texts, query=False, batch=64, max_length=512):
        if query:
            texts = [f"Instruct: {TASK}\nQuery:{t}" for t in texts]
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tokenizer(texts[i:i + batch], padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt").to("cuda")
                last = self.model(**enc).last_hidden_state[:, -1]
                out.append(self.torch.nn.functional.normalize(last.float(), dim=-1).cpu())
        return self.torch.cat(out).numpy()


@cache
def embedder():
    return Embedder()


@cache
def pool_vectors(dataset):
    """Paragraph embeddings, computed once and cached on disk."""
    path = CACHE / f"{dataset}-{EMBEDDER[0].split('/')[1]}-{EMBEDDER[1][:8]}.npy"
    if path.exists():
        vectors = np.load(path)
        assert len(vectors) == len(pool(dataset)), path
        return vectors
    CACHE.mkdir(parents=True, exist_ok=True)
    texts = [p["title"] + "\n" + p["paragraph_text"] for p in pool(dataset)]
    vectors = embedder()(texts).astype(np.float32)
    np.save(path, vectors)
    return vectors


def dense(dataset, queries, k=5):
    """Top-k pool ids per query, best first."""
    scores = embedder()(list(queries), query=True) @ pool_vectors(dataset).T
    return [list(np.argsort(-row)[:k]) for row in scores]


# ------------------------------------------------------------------- bm25
def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(d)) for d in documents]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lengths.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b
        self.postings = {}
        for i, d in enumerate(self.docs):
            for t, f in d.items():
                self.postings.setdefault(t, []).append((i, f))

    def scores(self, query):
        out = np.zeros(len(self.docs))
        for t in set(tokens(query)):
            for i, f in self.postings.get(t, ()):
                norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                out[i] += self.idf[t] * f * (self.k1 + 1) / norm
        return out


@cache
def bm25_index(dataset):
    return BM25([p["title"] + " " + p["paragraph_text"] for p in pool(dataset)])


def bm25(dataset, queries, k=5):
    index = bm25_index(dataset)
    return [list(np.argsort(-index.scores(q))[:k]) for q in queries]


def retrieve(dataset, queries, k=5, method="dense"):
    return (dense if method == "dense" else bm25)(dataset, queries, k)


def passages(dataset, pids):
    """Pool paragraphs in the row format the prompts use (idx, title, paragraph_text)."""
    rows = pool(dataset)
    return [{"idx": int(pid), "title": rows[pid]["title"],
             "paragraph_text": rows[pid]["paragraph_text"]} for pid in pids]


if __name__ == "__main__":
    for name in ("phantomwiki", "musique"):
        print(name, len(pool(name)), "paragraphs;", pool_vectors(name).shape)
