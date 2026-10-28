"""Checks for the retrieval contexts (run.py "__<context>" methods), with no model calls
and no GPU: retrieval is replaced by a deterministic stand-in and every log is in memory.

    python -m unittest llm_prompting.test_rag
"""

import hashlib
import json
import unittest
from pathlib import Path

from llm_prompting import prompts, retrieval, run
from llm_prompting.test_aggregate import FakeModel

ROW = json.loads(Path("data/cs470_dev48.jsonl").read_text().splitlines()[0])
PLANS = run.CallLog("logs/cs470/plans_deepseek.calls.jsonl")


def fake_retrieve(pool, queries, k=5, method="dense"):
    """Pool ids derived from each query's text, so a test can predict them."""
    out = []
    for q in queries:
        h = int(hashlib.sha256(q.encode()).hexdigest()[:8], 16)
        out.append([(h + i) % 1000 for i in range(k)])
    return out


class MemoryLog:
    """A CallLog kept in memory: records calls like CallLog and remembers prompts."""

    def __init__(self):
        self.done, self.messages, self.path = {}, {}, Path("<memory>")

    def call(self, model, key, messages, max_tokens):
        if key in self.done:
            return self.done[key]
        record = {"key": key, **model.generate(key, messages, max_tokens),
                  "prompt_sha256": hashlib.sha256(json.dumps(
                      messages, sort_keys=True).encode()).hexdigest()}
        self.done[key], self.messages[key] = record, messages
        return record


def shown(messages):
    """The passage ids a prompt shows."""
    return [p["idx"] for p in json.loads(messages[1]["content"].split("\n")[0])["passages"]]


class Contexts(unittest.TestCase):
    def setUp(self):
        self.saved = retrieval.retrieve
        retrieval.retrieve = fake_retrieve
        run._models.update(qwen=FakeModel(), deepseek=FakeModel())

    def tearDown(self):
        retrieval.retrieve = self.saved
        run._models.clear()

    def test_names(self):
        self.assertEqual(run.split_context("small__ragq"), ("small", "ragq"))
        self.assertEqual(run.split_context("deepseek_plan_commit__rag_final"),
                         ("deepseek_plan_commit", "rag_final"))
        self.assertEqual(run.split_context("guided_gold"), ("guided_gold", None))
        self.assertIn("gold_plan_deepseek__rag", run.METHODS)
        self.assertIn("guided_deepseek_plan_commit__ragsub", run.METHODS)
        self.assertNotIn("small__rag", run.METHODS)  # chains only
        self.assertNotIn("deepseek_plan__ragq", run.METHODS)

    def test_direct_sees_the_questions_passages(self):
        for ctx, k in (("ragq", 5), ("ragq15", 15)):
            log = MemoryLog()
            _, info = run.answer_directly(ROW, "qwen", log, 0, ctx=ctx)
            want = fake_retrieve("musique", [run.query_text(ROW["question"], ROW)], k)[0]
            self.assertEqual(info["retrieved"], want)
            self.assertEqual(shown(next(iter(log.messages.values()))), want)

    def test_guided_ragsub_sees_the_union_of_sub_questions(self):
        log = MemoryLog()
        _, info = run.answer_guided(ROW, "deepseek", "qwen", PLANS, log, 0, ctx="ragsub")
        union = []
        for hits in fake_retrieve("musique", [run.query_text(s, ROW) for s in info["plan"]]):
            union += [p for p in hits if p not in union]
        self.assertEqual(info["retrieved"], union)
        self.assertEqual(shown(next(iter(log.messages.values()))), union)

    def test_chain_rag_steps_retrieve_their_own_filled_question(self):
        log = MemoryLog()
        _, info = run.answer_with_plan(ROW, "gold", "qwen", None, log, 0, ctx="rag")
        for i, hop in enumerate(info["hops"]):
            want = fake_retrieve("musique", [run.query_text(hop["question"], ROW)])[0]
            self.assertEqual(hop["retrieved"], want)
            self.assertEqual(shown(log.messages[f"{ROW['id']}/rep0/hop{i}"]), want)
        self.assertNotIn("#1", info["hops"][-1]["question"])  # earlier answers filled in

    def test_rag_final_reuses_the_steps_and_reads_their_union(self):
        steps = MemoryLog()
        run.answer_with_plan(ROW, "gold", "qwen", None, steps, 0, ctx="rag")
        log = MemoryLog()
        _, info = run.answer_with_plan(ROW, "gold", "qwen", None, log, 0, ctx="rag_final",
                                       step_log=steps)
        self.assertEqual(list(log.messages), [f"{ROW['id']}/rep0/aggregate"])  # no new steps
        union = []
        for hop in info["hops"]:
            union += [p for p in hop["retrieved"] if p not in union]
        self.assertEqual(info["retrieved"], union)
        final = log.messages[f"{ROW['id']}/rep0/aggregate"]
        self.assertEqual(shown(final), union)
        self.assertIn('"question": ' + json.dumps(ROW["question"]), final[1]["content"])

    def test_commit_changes_only_the_instruction(self):
        plain, commit = MemoryLog(), MemoryLog()
        run.answer_with_plan(ROW, "gold", "qwen", None, plain, 0, ctx="rag")
        run.answer_with_plan(ROW, "gold", "qwen", None, commit, 0, ctx="rag", commit=True)
        key = f"{ROW['id']}/rep0/hop0"
        self.assertTrue(plain.messages[key][1]["content"].endswith(prompts.REASON))
        self.assertTrue(commit.messages[key][1]["content"].endswith(prompts.COMMIT))
        self.assertEqual(shown(plain.messages[key]), shown(commit.messages[key]))

    def test_given_context_chain_is_unchanged(self):
        """The given-context chain's first step is the saved prompt exactly. (Later steps
        fill #k with the stub's answers; test_aggregate replays every saved step with the
        real answers.)"""
        saved = run.CallLog("logs/cs470/deepseek_plan_rep0.calls.jsonl")
        log = MemoryLog()
        run.answer_with_plan(ROW, "deepseek", "qwen", PLANS, log, 0)
        key = f"{ROW['id']}/rep0/hop0"
        self.assertEqual(log.done[key]["prompt_sha256"], saved.done[key]["prompt_sha256"])


if __name__ == "__main__":
    unittest.main()
