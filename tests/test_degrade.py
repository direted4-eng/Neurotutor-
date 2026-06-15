"""529 graceful-degrade + re-grade: a MiniMax overload must DELAY feedback,
never drop the answer. This pins the whole save→defer→regrade cycle end-to-end
against a temp DB (no network — the grader is mocked).
"""
import unittest
from unittest.mock import patch

from neurotutor.db import store
from neurotutor.agent import drill

UID = "u-degrade-test"


def _insert_open_question() -> int:
    # concept_id=NULL → scenario-style: graded & logged but no FSRS scheduling,
    # which keeps this test focused on the degrade/regrade path.
    with store.connect() as conn:
        cur = conn.execute(
            """INSERT INTO pending_questions(user_id, concept_id, bloom_level,
                                             prompt, rubric, kind, topic)
               VALUES (?,?,?,?,?,?,?)""",
            (UID, None, 3, "Опиши тактику при разрыве аневризмы.",
             None, "emergency", "Разрыв аневризмы"))
        return cur.lastrowid


class TestGracefulDegrade(unittest.TestCase):
    def setUp(self):
        store.init_db()  # idempotent; guarantees schema on the temp DB
        with store.connect() as conn:
            conn.execute("DELETE FROM pending_questions WHERE user_id=?", (UID,))
            conn.execute("DELETE FROM responses WHERE prompt LIKE 'Опиши тактику%'")
        self.qid = _insert_open_question()

    def _saved_answer(self):
        with store.connect() as conn:
            row = conn.execute(
                "SELECT saved_answer, answered_at FROM pending_questions WHERE id=?",
                (self.qid,)).fetchone()
        return row["saved_answer"], row["answered_at"]

    def test_overload_defers_then_regrade_completes(self):
        # 1) MiniMax overloaded → grade_answer raises. Answer must be saved,
        #    question stays open, caller gets a degraded signal (not an exception).
        with patch("neurotutor.agent.drill.grade_answer",
                   side_effect=RuntimeError("transient 529")):
            fb = drill.answer_pending(UID, "клипирование/койлинг, контроль ВЧД")
        self.assertTrue(fb and fb.get("degraded"))
        saved, answered = self._saved_answer()
        self.assertEqual(saved, "клипирование/койлинг, контроль ВЧД")
        self.assertIsNone(answered)
        self.assertTrue(drill.has_open(UID))               # still open
        self.assertEqual(len(drill.pending_regrade(UID)), 1)

        # 2) MiniMax recovers → re-grade pass grades the saved answer for real.
        good = {"score": 0.8, "suggested_rating": 3, "feedback": "ок",
                "breakdown": {"x": 0.8}}
        with patch("neurotutor.agent.drill.grade_answer", return_value=good):
            results = drill.regrade_saved(UID)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["score"], 0.8)
        saved, answered = self._saved_answer()
        self.assertIsNone(saved)                            # cleared
        self.assertIsNotNone(answered)                      # now graded
        self.assertFalse(drill.has_open(UID))               # closed
        self.assertEqual(len(drill.pending_regrade(UID)), 0)

    def test_regrade_noop_when_nothing_deferred(self):
        self.assertEqual(drill.regrade_saved(UID), [])


if __name__ == "__main__":
    unittest.main()
