"""esc(): dynamic LLM/user text must be HTML-safe before it lands in a
parse_mode=HTML message, or a stray '<' 400s the whole send and the feedback is
lost silently (the 'bot goes quiet' bug).
"""
import unittest

import tg_handler as t


class TestEsc(unittest.TestCase):
    def test_escapes_angles_and_amp(self):
        self.assertEqual(t.esc("ВЧД < 20 & ВЧГ"), "ВЧД &lt; 20 &amp; ВЧГ")

    def test_escapes_closing_tag_injection(self):
        self.assertEqual(t.esc("</b> oops <i>"), "&lt;/b&gt; oops &lt;i&gt;")

    def test_none_is_empty(self):
        self.assertEqual(t.esc(None), "")

    def test_quotes_preserved(self):
        # quote=False: don't mangle quotes inside «...» captions
        self.assertEqual(t.esc('диагноз "САК"'), 'диагноз "САК"')

    def test_plain_text_unchanged(self):
        self.assertEqual(t.esc("аневризма ПСА"), "аневризма ПСА")


if __name__ == "__main__":
    unittest.main()
