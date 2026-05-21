"""External authoritative medical sources.

PubMed via E-utilities is the primary; we also expose helpers for
Cochrane and guidelines via PubMed filters (review[pt], guideline[pt]).
Rate-limited per NCBI policy (3/s without key, 10/s with key).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from ..config import SETTINGS

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    min_gap = 0.1 if SETTINGS.pubmed_api_key else 0.34
    wait = min_gap - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _params(extra: dict[str, Any]) -> dict[str, Any]:
    p = {"db": "pubmed", "tool": "neurotutor"}
    if SETTINGS.pubmed_email:
        p["email"] = SETTINGS.pubmed_email
    if SETTINGS.pubmed_api_key:
        p["api_key"] = SETTINGS.pubmed_api_key
    p.update(extra)
    return p


def pubmed_search(query: str, *, max_results: int = 5,
                  filter_: str | None = None) -> list[dict]:
    """Search PubMed; return list of {pmid, title, abstract, journal, year}."""
    term = query
    if filter_:
        term = f"({query}) AND {filter_}"

    _throttle()
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{EUTILS}/esearch.fcgi",
                       params=_params({"term": term, "retmode": "json",
                                       "retmax": max_results}))
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        _throttle()
        r = client.get(f"{EUTILS}/efetch.fcgi",
                       params=_params({"id": ",".join(ids), "rettype": "abstract",
                                       "retmode": "xml"}))
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "xml")
    out: list[dict] = []
    for art in soup.find_all("PubmedArticle"):
        pmid = (art.find("PMID").text if art.find("PMID") else "").strip()
        title = (art.find("ArticleTitle").text if art.find("ArticleTitle") else "")
        abstract = " ".join(
            (a.text or "") for a in art.find_all("AbstractText")
        ).strip()
        journal = (art.find("Title").text if art.find("Title") else "")
        year = ""
        pd = art.find("PubDate")
        if pd and pd.find("Year"):
            year = pd.find("Year").text
        out.append({
            "pmid": pmid, "title": title, "abstract": abstract,
            "journal": journal, "year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return out
