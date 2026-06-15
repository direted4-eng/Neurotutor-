"""Routing + trigger-regex tests — the fragile substring logic the audit flagged.

These are pure functions (no DB, no network), so they pin the exact behaviour
that grading/case flows depend on: a wrong route silently sends an answer to the
wrong handler.
"""
import unittest

import tg_handler as t


class TestRoute(unittest.TestCase):
    def test_reset_beats_everything(self):
        self.assertEqual(t.route("/reset")[0], "reset")
        self.assertEqual(t.route("стоп")[0], "reset")
        # reset must win even alongside a mode keyword
        self.assertEqual(t.route("новая тема, давай кейс")[0], "reset")

    def test_persona(self):
        self.assertEqual(t.route("персонаж корвин"), ("persona:corvin", "corvin"))
        self.assertEqual(t.route("май линь"), ("persona:lin", "lin"))

    def test_modes(self):
        self.assertEqual(t.route("проверь меня")[0], "diagnostic")
        self.assertEqual(t.route("что повторить")[0], "review")
        self.assertEqual(t.route("разбери клинический случай")[0], "case")
        self.assertEqual(t.route("давай экзамен")[0], "osce")
        self.assertEqual(t.route("покажи кт")[0], "imaging")
        self.assertEqual(t.route("расскажи про аневризмы")[0], "new")

    def test_unknown_keeps_mode(self):
        # empty routing = keep current mode (the answer-to-open-question path)
        self.assertEqual(t.route("внутренняя сонная артерия даёт глазную"), ("", ""))

    def test_short_keyword_not_matched_inside_word(self):
        # regression: 'лин' must NOT match inside 'клинический' → persona:lin
        self.assertNotEqual(t.route("клинический разбор")[0], "persona:lin")
        # but the actual persona name 'линь' still works as a whole word
        self.assertEqual(t.route("давай как персонаж линь"), ("persona:lin", "lin"))
        # and 'кт' must not fire imaging from inside a longer word
        self.assertEqual(t.route("расскажи про контакт нерва")[0], "new")


class TestFinalRe(unittest.TestCase):
    def test_bare_conclusion_has_empty_rest(self):
        m = t.FINAL_RE.match("заключение")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).strip(), "")

    def test_inline_conclusion_captures_rest(self):
        m = t.FINAL_RE.match("заключение: аневризма ПСА, клипирование")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).strip(), "аневризма ПСА, клипирование")

    def test_my_diagnosis_variant(self):
        self.assertIsNotNone(t.FINAL_RE.match("мой диагноз — САК"))

    def test_does_not_eat_workup_prose(self):
        # "итог обследования…" must NOT be treated as the final answer trigger
        self.assertIsNone(t.FINAL_RE.match("итог обследования: нужна ангиография"))


class TestCancelRe(unittest.TestCase):
    def test_matches(self):
        for s in ("отмена", "назад", "продолжим разбор", "погоди"):
            self.assertIsNotNone(t.CANCEL_RE.match(s), s)

    def test_no_false_positive(self):
        self.assertIsNone(t.CANCEL_RE.match("аневризма передней соединительной"))


class TestConspectRe(unittest.TestCase):
    def test_topic_extraction(self):
        m = t.CONSPECT_RE.match("конспект по аневризмам ПСА")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).strip(), "аневризмам ПСА")

    def test_theory_synonym(self):
        m = t.CONSPECT_RE.match("сделай теорию о шунтах")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).strip(), "шунтах")


if __name__ == "__main__":
    unittest.main()
