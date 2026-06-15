"""_strip_scales: severity-scale gradings betray the diagnosis category, so a
case prompt must hide them — while keeping generic findings (GCS, vitals) that
don't give the answer away.
"""
import unittest

from neurotutor.agent.drill import _strip_scales


class TestStripScales(unittest.TestCase):
    def test_removes_diagnosis_revealing_scales(self):
        for blurb in (
            "Пациент с Hunt-Hess 3 после внезапной головной боли",
            "Опухоль WHO grade IV в левой лобной доле",
            "Кровоизлияние, Fisher 4 по КТ",
            "АВМ, Spetzler-Martin III",
            "Травма спинного мозга, ASIA B",
        ):
            out = _strip_scales(blurb)
            low = out.lower()
            for tell in ("hunt", "who grade", "fisher", "spetzler", "asia"):
                self.assertNotIn(tell, low, f"{tell!r} leaked from {blurb!r}: {out!r}")

    def test_keeps_generic_findings(self):
        out = _strip_scales("GCS 14, АД 150/90, зрачки D=S, очаговой симптоматики нет")
        self.assertIn("GCS 14", out)
        self.assertIn("150/90", out)

    def test_no_double_spaces_or_dangling_punct(self):
        out = _strip_scales("Пациент с Hunt-Hess 3 , далее осмотр")
        self.assertNotIn("  ", out)
        self.assertNotIn(" ,", out)


if __name__ == "__main__":
    unittest.main()
