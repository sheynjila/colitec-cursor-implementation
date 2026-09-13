from __future__ import annotations

from dataclasses import dataclass

import httpx

from aar.config import Settings
from aar.errors import AARError
from aar.textutil import strip_html
from aar.tools.ssrf import UrlBlocked, fetch_text, validate_url

FIXTURE_PAGES = {
    "https://example.com/fixture/research-safety": {
        "title": "[fixture] Research safety constraints",
        "text": (
            "Evidence-grounded research assistants must validate every candidate fact "
            "against retrieved evidence. Coverage retries are limited to two total "
            "research rounds, and uncovered sub-questions must be disclosed as "
            "limitations rather than filled with fabricated citations."
        ),
    },
    "https://example.com/fixture/routing-budget": {
        "title": "[fixture] Routing and budget",
        "text": (
            "Model routing must prefer the lowest projected cost among eligible aliases "
            "in the required tier. A zero-budget run must make zero paid model calls."
        ),
    },
}


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    labeled_fixture: bool = False


class WebSearchTool:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        if self.settings.web_mode == "fixture" or self.settings.demo_fixtures:
            return _fixture_hits(query, limit)
        if not self.settings.tavily_api_key:
            raise AARError("web_unavailable", "Tavily is not configured; web search was skipped.")
        return _tavily_search(self.settings.tavily_api_key, query, limit)

    def fetch(self, url: str) -> tuple[str, str, bool]:
        if url in FIXTURE_PAGES and (
            self.settings.web_mode == "fixture" or self.settings.demo_fixtures
        ):
            page = FIXTURE_PAGES[url]
            return url, page["text"], True
        validate_url(url)
        final_url, raw = fetch_text(
            url,
            timeout=self.settings.worker_timeout_seconds,
            max_bytes=self.settings.fetch_max_bytes,
        )
        return final_url, strip_html(raw), False


def _fixture_hits(query: str, limit: int) -> list[SearchHit]:
    hits = []
    for url, page in FIXTURE_PAGES.items():
        blob = f"{page['title']} {page['text']} {query}".lower()
        if any(token in blob for token in query.lower().split() if len(token) > 3):
            hits.append(
                SearchHit(
                    url=url,
                    title=page["title"],
                    snippet=page["text"][:180],
                    labeled_fixture=True,
                )
            )
    if not hits:
        url, page = next(iter(FIXTURE_PAGES.items()))
        hits.append(
            SearchHit(
                url=url,
                title=page["title"],
                snippet=page["text"][:180],
                labeled_fixture=True,
            )
        )
    return hits[:limit]


def _tavily_search(api_key: str, query: str, limit: int) -> list[SearchHit]:
    response = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": limit},
        timeout=20.0,
    )
    response.raise_for_status()
    data = response.json()
    hits: list[SearchHit] = []
    for item in data.get("results") or []:
        url = str(item.get("url") or "")
        try:
            validate_url(url)
        except UrlBlocked:
            continue
        hits.append(
            SearchHit(
                url=url,
                title=str(item.get("title") or url),
                snippet=str(item.get("content") or ""),
                labeled_fixture=False,
            )
        )
    return hits
