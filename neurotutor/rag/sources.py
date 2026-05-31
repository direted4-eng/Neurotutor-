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
RADIOPAEDIA_SEARCH = "https://radiopaedia.org/search"


def radiopaedia_search(query: str, *, max_results: int = 5,
                       scope: str = "cases") -> list[dict]:
    """Search Radiopaedia for radiology cases or articles.

    scope: 'cases' | 'articles' | 'all'
    Returns list of {title, url, image_url, description, modality}.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Neurotutor/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    params = {"lang": "us", "q": query, "scope": scope}
    out: list[dict] = []
    try:
        with httpx.Client(timeout=15.0, headers=headers,
                          follow_redirects=True) as client:
            r = client.get(RADIOPAEDIA_SEARCH, params=params)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")

            # Results are in <div class="search-result"> blocks
            for card in soup.select(".search-result")[:max_results]:
                title_el = card.select_one(".search-result-title, h3, h4, .title")
                link_el = card.select_one("a[href]")
                desc_el = card.select_one(".search-result-description, .excerpt, p")
                img_el = card.select_one("img[src]")

                title = title_el.get_text(strip=True) if title_el else ""
                href = link_el["href"] if link_el else ""
                if href and not href.startswith("http"):
                    href = f"https://radiopaedia.org{href}"
                desc = desc_el.get_text(strip=True) if desc_el else ""
                img_url = ""
                if img_el:
                    img_url = img_el.get("src") or img_el.get("data-src") or ""

                # Try to detect modality from title/description
                text_lower = (title + " " + desc).lower()
                modality = "unknown"
                for m in ("mri", "ct", "angiograph", "x-ray", "pet", "spect",
                          "ultrasound", "dsa"):
                    if m in text_lower:
                        modality = m.upper()
                        break

                if title or href:
                    out.append({
                        "title": title,
                        "url": href,
                        "image_url": img_url,
                        "description": desc[:300],
                        "modality": modality,
                    })
    except Exception as exc:
        log.warning("radiopaedia_search failed: %s", exc)
    return out
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
                  filter_: str | None = None,
                  sort: str | None = None,
                  reldate: int | None = None) -> list[dict]:
    """Search PubMed; return list of {pmid, title, abstract, journal, year}.

    sort: e.g. 'date' (most recent first) or 'relevance'.
    reldate: restrict to the last N days (datetype=pdat).
    """
    term = query
    if filter_:
        term = f"({query}) AND {filter_}"

    extra: dict[str, Any] = {"term": term, "retmode": "json",
                             "retmax": max_results}
    if sort:
        extra["sort"] = sort
    if reldate:
        extra["reldate"] = reldate
        extra["datetype"] = "pdat"

    _throttle()
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{EUTILS}/esearch.fcgi", params=_params(extra))
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
