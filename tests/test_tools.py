from __future__ import annotations

import socket

import pytest

from aar.tools.ssrf import UrlBlocked, validate_url
from aar.tools.web import WebSearchTool


def test_ssrf_rejects_loopback_and_private(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_addr(host: str, _port: object) -> list:
        mapping = {
            "localhost": "127.0.0.1",
            "internal.local": "10.0.0.8",
            "link.local": "169.254.1.1",
        }
        ip = mapping.get(host, "8.8.8.8")
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    monkeypatch.setattr("aar.tools.ssrf.socket.getaddrinfo", fake_addr)
    with pytest.raises(UrlBlocked):
        validate_url("http://localhost/secret")
    with pytest.raises(UrlBlocked):
        validate_url("http://internal.local/admin")
    with pytest.raises(UrlBlocked):
        validate_url("http://link.local/meta")
    with pytest.raises(UrlBlocked):
        validate_url("file:///etc/passwd")
    assert validate_url("https://example.com/ok")


def test_fixture_search_is_labeled(settings) -> None:
    tool = WebSearchTool(settings)
    hits = tool.search("coverage retries")
    assert hits
    assert all(hit.labeled_fixture for hit in hits)
    url, text, labeled = tool.fetch(hits[0].url)
    assert labeled
    assert "two total research rounds" in text or text
    assert url.startswith("https://example.com/fixture/")
