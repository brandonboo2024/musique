"""Checks for the guided, gold_plan and cap4096 methods (run.py), with no model calls.

    python -m unittest llm_prompting.test_guided
"""

import json
import unittest
from pathlib import Path

from llm_prompting import prompts, run
from llm_prompting.test_aggregate import FakeModel, _MemoryLog

DATA = (("", "data/cs470_dev48.jsonl"), ("phantomwiki48", "data/cs470_phantomwiki48.jsonl"))


def rows(gold):
    return [json.loads(line) for line in Path(gold).read_text().splitlines()]


class GoldPlan(unittest.TestCase):
    def test_every_gold_plan_parses_as_a_plan(self):
        for _, gold in DATA:
            for row in rows(gold):
                steps = run.gold_plan(row)
                text = json.dumps({"subquestions": steps})
                self.assertEqual(prompts.parse_plan(text, row.get("max_steps", 4)), steps)

    def test_phantomwiki_plan_follows_the_goals(self):
        row = rows(DATA[1][1])[0]
        self.assertEqual(run.gold_plan(row), [
            "Who is the person whose date of birth is 1006-07-10?",
            "Who is the friend of #1?", "What is the hobby of #2?"])


class Methods(unittest.TestCase):
    def test_names(self):
        self.assertEqual(run.split_method("deepseek_plan_cap4096_hidden"),
                         ("deepseek_plan_cap4096", "hidden", None))
        self.assertEqual(run.PLANNED["small_plan_cap4096"], ("qwen", "qwen"))
        self.assertEqual(run.PLANNED["gold_plan"], ("gold", "qwen"))
        self.assertEqual(run.GUIDED["guided_gold_deepseek"], ("gold", "deepseek"))
        self.assertTrue(set(run.GUIDED) <= set(run.METHODS))

    def test_guided_uses_every_saved_plan_in_one_call(self):
        run._models.update(qwen=FakeModel(), deepseek=FakeModel())
        for dataset, gold in DATA:
            logs = Path("logs/cs470") / dataset
            for method, (planner, answerer) in run.GUIDED.items():
                if planner == "deepseek_rev":
                    continue  # draws new revisions; covered by test_revision_falls_back
                plan_log = (None if planner == "gold"
                            else run.CallLog(logs / f"plans_{planner}.calls.jsonl"))
                for row in rows(gold):
                    log = _MemoryLog()
                    answer, info = run.answer_guided(row, planner, answerer, plan_log, log, 0)
                    if info["plan"] is None:
                        continue  # an unusable saved plan; no call is made
                    self.assertEqual(answer, "x")
                    self.assertEqual(len(log.messages), 1)
                    data = json.loads(log.messages[0][1]["content"].split("\n")[0])
                    self.assertEqual(data["question"], row["question"])
                    self.assertEqual(data["plan"], info["plan"])
        run._models.clear()

    def test_revision_falls_back_to_the_saved_plan(self):
        """A revision that does not parse keeps DeepSeek's saved plan. The revision
        log is in memory, so no real log is written."""
        row = rows(DATA[0][1])[0]

        class RevLog(_MemoryLog):
            def __init__(self):
                super().__init__()
                self.path = Path("logs/cs470/plans_deepseek_rev.calls.jsonl")  # its folder

        real = Path("logs/cs470/plans_deepseek_rev.calls.jsonl")
        before = real.read_bytes() if real.exists() else None
        run._models.update(deepseek=FakeModel(), qwen=FakeModel())
        log = _MemoryLog()
        answer, info = run.answer_guided(row, "deepseek_rev", "qwen", RevLog(), log, 0)
        run._models.clear()
        saved = run.CallLog("logs/cs470/plans_deepseek.calls.jsonl").done[f"{row['id']}/plan"]
        self.assertEqual(info["plan"], prompts.parse_plan(saved["text"]))
        self.assertEqual(info["cost"]["calls"], 3)  # saved plan, revision, answer
        self.assertEqual(real.read_bytes() if real.exists() else None, before)

    def test_cap_reuses_the_saved_plan_and_raises_the_step_cap(self):
        row = rows(DATA[0][1])[0]
        caps = []

        class Log(_MemoryLog):
            def call(self, model, key, messages, max_tokens):
                caps.append(max_tokens)
                return super().call(model, key, messages, max_tokens)

        run._models.update(qwen=FakeModel())
        plan_log = run.CallLog("logs/cs470/plans_deepseek.calls.jsonl")
        run.answer_with_plan(row, "deepseek", "qwen", plan_log, Log(), 0,
                             hop_cap=run.DIRECT_CAP)
        run._models.clear()
        self.assertTrue(caps and set(caps) == {run.DIRECT_CAP})


if __name__ == "__main__":
    unittest.main()
