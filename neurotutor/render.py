"""Render Markdown study notes into a clean, readable PDF.

The LLM produces Markdown; raw Markdown is unreadable on a phone (`#`, `**`).
We convert it to styled HTML and let weasyprint (system CLI) lay it out as a
polished A4 handout. Palette matches the Live-maps dashboard for cohesion.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile

import markdown as _md

log = logging.getLogger(__name__)

WEASYPRINT = "/usr/local/bin/weasyprint"

_CSS = """
@page { size: A4; margin: 22mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 11.5pt; line-height: 1.55; color: #2b2017;
}
.cover { margin-bottom: 18pt; border-bottom: 3px solid #4F9B43; padding-bottom: 10pt; }
.brand { font-family: "DejaVu Sans", sans-serif; font-size: 9pt;
         letter-spacing: 1px; text-transform: uppercase; color: #9B4F43; }
.cover-title { font-family: "DejaVu Sans", sans-serif; font-size: 22pt;
               color: #3D2E1F; margin: 4pt 0 0 0; line-height: 1.2; }
h1 { font-family: "DejaVu Sans", sans-serif; font-size: 16pt; color: #3D2E1F;
     margin: 16pt 0 6pt; }
h2 { font-family: "DejaVu Sans", sans-serif; font-size: 13.5pt; color: #9B4F43;
     border-bottom: 1px solid #e3d6c5; padding-bottom: 3pt; margin: 16pt 0 6pt; }
h3 { font-family: "DejaVu Sans", sans-serif; font-size: 11.5pt; color: #4F9B43;
     margin: 12pt 0 4pt; }
p { margin: 5pt 0; }
ul, ol { margin: 5pt 0 5pt 0; padding-left: 20pt; }
li { margin: 2.5pt 0; }
strong { color: #3D2E1F; }
blockquote { margin: 8pt 0; padding: 6pt 12pt; background: #f5efe6;
             border-left: 3px solid #AF8F6B; color: #4a3b2a; font-style: italic; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 10pt;
       background: #f0e9dd; padding: 0 3px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 10.5pt; }
th, td { border: 1px solid #d8c8b2; padding: 4pt 7pt; text-align: left; }
th { background: #efe0d0; }
hr { border: none; border-top: 1px solid #e3d6c5; margin: 12pt 0; }
"""

_TMPL = (
    '<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
    "<style>{css}</style></head><body>"
    '<div class="cover"><div class="brand">NeuroTutor · конспект</div>'
    '<div class="cover-title">{title}</div></div>{body}</body></html>'
)


def markdown_to_pdf(md_text: str, *, title: str = "Конспект") -> bytes | None:
    """Convert Markdown to a styled PDF. Returns bytes, or None on failure."""
    # Drop a leading top-level "# Heading" — the title is shown on the cover.
    md_text = re.sub(r"^\s*#\s+.*\n", "", md_text, count=1)
    body = _md.markdown(md_text, extensions=["extra", "sane_lists"])
    html = _TMPL.format(css=_CSS, title=title, body=body)

    with tempfile.TemporaryDirectory() as d:
        hp = os.path.join(d, "in.html")
        pp = os.path.join(d, "out.pdf")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(html)
        try:
            subprocess.run([WEASYPRINT, hp, pp], check=True,
                           capture_output=True, timeout=90)
            with open(pp, "rb") as f:
                return f.read()
        except Exception:
            log.exception("weasyprint render failed")
            return None
