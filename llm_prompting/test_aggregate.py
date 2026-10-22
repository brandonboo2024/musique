"""Checks for the hidden and context modes (run.py), with no model calls.

    python -m unittest llm_prompting.test_aggregate
"""

import json
import unittest
from pathlib import Path

from llm_prompting import prompts, run


class FakeModel:
    """Answers every call with a fixed line, so the aggregation path runs offline."""

    def generate(self, key, messages, max_tokens):
        return {"model": "fake", "text": "FINAL_ANSWER: x", "input_tokens": 0,
                "output_tokens": 1, "finish_reason": "stop", "seconds": 0, "usd": 0.0}


class ReadAnswer(unittest.TestCase):
    def test_inline_and_markdown(self):
        text = "Eldon is in Miller County, Missouri. FINAL_ANSWER: Miller County"
        self.assertEqual(prompts.read_answer(text), "Miller County")
        self.assertEqual(prompts.final_answer(text)[1], False)  # still flagged
        self.assertEqual(prompts.read_answer("x\n**FINAL_ANSWER:** Alvaro Smock"),
                         "Alvaro Smock")
        self.assertEqual(prompts.read_answer("x\nFINAL_ANSWER: 2**"), "2")
        self.assertEqual(prompts.read_answer("FINAL_ANSWER:\nUNKNOWN"), "UNKNOWN")
        self.assertEqual(prompts.read_answer("a\nFINAL_ANSWER: 1\nFINAL_ANSWER: 2"), "2")
        self.assertIsNone(prompts.read_answer("reasoning cut off at the token cap"))

    def test_step_record_marks_missing_answer_and_drops_note(self):
        note = " If there are several answers, list all of them separated by commas."
        rec = prompts.step_record("Who is the son of Gene Smock?" + note, "no line", note)
        self.assertEqual(rec, {"subquestion": "Who is the son of Gene Smock?",
                               "answer": "NO ANSWER (the step did not finish)"})


class Modes(unittest.TestCase):
    def test_method_names(self):
        self.assertEqual(run.split_method("small_plan_hidden"), ("small_plan", "hidden", None))
        self.assertEqual(run.split_method("deepseek_plan_deepseek_context"),
                         ("deepseek_plan_deepseek", "context", None))
        self.assertEqual(run.split_method("small_plan_context_dsread"),
                         ("small_plan", "context", "deepseek"))
        self.assertEqual(run.split_method("deepseek_plan"), ("deepseek_plan", "chain", None))
        self.assertNotIn("deepseek_plan_deepseek_hidden_dsread", run.PLANNED)
        self.assertEqual(run.PLANNED["deepseek_plan_deepseek_hidden"], ("deepseek", "deepseek"))

    def test_hidden_reuses_every_saved_step(self):
        """Every saved step of every planned method rebuilds to the same prompt hash."""
        for dataset, gold in (("", "data/cs470_dev48.jsonl"),
                              ("phantomwiki48", "data/cs470_phantomwiki48.jsonl")):
            logs = Path("logs/cs470") / dataset
            rows = [json.loads(line) for line in Path(gold).read_text().splitlines()]
            for base, (planner, answerer) in (("deepseek_plan", ("deepseek", "qwen")),
                                              ("small_plan", ("qwen", "qwen")),
                                              ("deepseek_plan_deepseek",
                                               ("deepseek", "deepseek"))):
                plan_log = run.CallLog(logs / f"plans_{planner}.calls.jsonl")
                for rep in (0, 1, 2):
                    base_log = run.CallLog(logs / f"{base}_rep{rep}.calls.jsonl")
                    log = _MemoryLog()
                    run._models.update(qwen=FakeModel(), deepseek=FakeModel())
                    for row in rows:
                        answer, info = run.answer_with_plan(
                            row, planner, answerer, plan_log, log, rep, "hidden", base_log)
                        if info["plan"] is not None:
                            self.assertEqual(answer, "x")
                            history = json.loads(log.messages[-1][1]["content"].split(
                                "\n")[0])["subquestion_answers"]
                            self.assertEqual(len(history), len(info["plan"]))
        run._models.clear()

    def test_context_step_sees_earlier_steps(self):
        row = json.loads(Path("data/cs470_dev48.jsonl").read_text().splitlines()[0])
        history = [{"subquestion": "Who is X?", "answer": "Y"}]
        user = prompts.context_hop_prompt("Who married that man?", history, row)[1]["content"]
        data = json.loads(user.split("\n")[0])
        self.assertEqual(data["question"], "Who married that man?")
        self.assertEqual(data["earlier_subquestion_answers"], history)
        agg = json.loads(prompts.aggregate_prompt(row, history)[1]["content"].split("\n")[0])
        self.assertEqual(agg["question"], row["question"])


class _MemoryLog:
    """A CallLog that keeps calls in memory and remembers their prompts."""

    def __init__(self):
        self.messages, self.path = [], Path("<memory>")

    def call(self, model, key, messages, max_tokens):
        self.messages.append(messages)
        return {"key": key, **model.generate(key, messages, max_tokens)}


if __name__ == "__main__":
    unittest.main()
