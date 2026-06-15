"""grade_answer + _rating_from_score — the honesty-critical grading core.

The LLM is mocked (a fake MiniMaxClient returns a canned chat response), so these
run with no network and pin: score = mean of per-criterion marks, deterministic
FSRS rating, and the ungraded (parse-error) path that must NOT fabricate a 0.
"""
import json
import unittest
from unittest.mock import patch

from neurotutor.agent.tools import grade_answer, _rating_from_score


class FakeClient:
    """Stand-in for MiniMaxClient: .chat() returns a chat-completion-shaped dict
    whose content is whatever raw string the test wants the grader to 'see'."""
    def __init__(self, content: str):
        self._content = content

    def chat(self, *args, **kwargs):
        return {"choices": [{"message": {"content": self._content}}]}

    def close(self):
        pass


class CyclingClient:
    """Returns a different canned content on each successive .chat() call —
    lets a test exercise self-consistency across genuinely differing samples."""
    def __init__(self, contents):
        self._contents = list(contents)
        self._i = 0

    def chat(self, *args, **kwargs):
        c = self._contents[self._i % len(self._contents)]
        self._i += 1
        return {"choices": [{"message": {"content": c}}]}

    def close(self):
        pass


def _grade_with(content: str, **kw):
    # Patch the symbol where grade_answer looks it up (inside neurotutor.llm.minimax).
    with patch("neurotutor.llm.minimax.MiniMaxClient", return_value=FakeClient(content)):
        return grade_answer(prompt="p", answer="a", **kw)


def _grade_cycling(contents, **kw):
    with patch("neurotutor.llm.minimax.MiniMaxClient",
               return_value=CyclingClient(contents)):
        return grade_answer(prompt="p", answer="a", **kw)


class TestRatingFromScore(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(_rating_from_score(0.90), 4)
        self.assertEqual(_rating_from_score(0.85), 4)
        self.assertEqual(_rating_from_score(0.84), 3)
        self.assertEqual(_rating_from_score(0.60), 3)
        self.assertEqual(_rating_from_score(0.59), 2)
        self.assertEqual(_rating_from_score(0.40), 2)
        self.assertEqual(_rating_from_score(0.39), 1)
        self.assertEqual(_rating_from_score(0.0), 1)


class TestGradeAnswer(unittest.TestCase):
    def test_score_is_mean_of_breakdown(self):
        content = json.dumps({"breakdown": {"a": 1.0, "b": 0.5}, "feedback": "ok"})
        g = _grade_with(content, bloom_level=1, rubric={"criteria": ["a", "b"]})
        self.assertAlmostEqual(g["score"], 0.75)
        self.assertEqual(g["suggested_rating"], 3)  # 0.75 -> good
        self.assertFalse(g.get("parse_error"))

    def test_score_clamped(self):
        content = json.dumps({"breakdown": {"a": 1.5, "b": 1.0}})
        g = _grade_with(content, bloom_level=1, rubric={"criteria": ["a", "b"]})
        self.assertLessEqual(g["score"], 1.0)

    def test_parse_error_is_ungraded_not_zero(self):
        g = _grade_with("the model rambled without any json here")
        self.assertIsNone(g["score"])
        self.assertTrue(g["parse_error"])

    def test_handles_json_fence(self):
        content = "```json\n" + json.dumps({"breakdown": {"x": 1.0}}) + "\n```"
        g = _grade_with(content, bloom_level=1, rubric={"criteria": ["x"]})
        self.assertAlmostEqual(g["score"], 1.0)


class TestSelfConsistency(unittest.TestCase):
    def test_median_across_samples(self):
        contents = [
            json.dumps({"breakdown": {"a": 0.6}, "feedback": "f06"}),
            json.dumps({"breakdown": {"a": 1.0}, "feedback": "f10"}),
            json.dumps({"breakdown": {"a": 0.8}, "feedback": "f08"}),
        ]
        g = _grade_cycling(contents, bloom_level=1, rubric={"criteria": ["a"]},
                           samples=3)
        self.assertEqual(g["samples_used"], 3)
        self.assertAlmostEqual(g["breakdown"]["a"], 0.8)   # median of .6/1/.8
        self.assertAlmostEqual(g["score"], 0.8)
        # representative feedback comes from the sample nearest the aggregate
        self.assertEqual(g["feedback"], "f08")

    def test_median_is_robust_to_one_outlier(self):
        contents = [
            json.dumps({"breakdown": {"a": 1.0, "b": 1.0}}),
            json.dumps({"breakdown": {"a": 1.0, "b": 1.0}}),
            json.dumps({"breakdown": {"a": 0.0, "b": 0.0}}),   # noisy outlier
        ]
        g = _grade_cycling(contents, bloom_level=1, rubric={"criteria": ["a", "b"]},
                           samples=3)
        # median ignores the outlier → 1.0, where a plain mean would be ~0.67
        self.assertAlmostEqual(g["score"], 1.0)

    def test_single_sample_path(self):
        content = json.dumps({"breakdown": {"a": 1.0, "b": 0.5}})
        g = _grade_cycling([content], bloom_level=1,
                           rubric={"criteria": ["a", "b"]}, samples=1)
        self.assertEqual(g["samples_used"], 1)
        self.assertAlmostEqual(g["score"], 0.75)


class TestCriterionWeights(unittest.TestCase):
    def test_weight_shifts_score(self):
        # core criterion nailed, minor missed; weight 3:1 → 0.75, not a flat 0.5
        content = json.dumps({"breakdown": {"core": 1.0, "minor": 0.0}})
        rubric = {"criteria": [{"name": "core", "weight": 3},
                               {"name": "minor", "weight": 1}]}
        g = _grade_cycling([content], rubric=rubric, samples=1)
        self.assertAlmostEqual(g["score"], 0.75)

    def test_equal_weights_match_plain_mean(self):
        content = json.dumps({"breakdown": {"a": 1.0, "b": 0.0}})
        g = _grade_cycling([content], rubric={"criteria": ["a", "b"]}, samples=1)
        self.assertAlmostEqual(g["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
