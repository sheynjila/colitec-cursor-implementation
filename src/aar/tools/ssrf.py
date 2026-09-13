"""SSRF checks for original URLs, redirects, and resolved destinations."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx

from aar.errors import AARError


class UrlBlocked(AARError):
    def __init__(self, message: str) -> None:
        super().__init__("unsafe_url", message)


_BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}


def _host_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        literal = ipaddress.ip_address(host)
        return [literal]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlBlocked(f"Hostname could not be resolved: {host}") from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        addresses.append(ipaddress.ip_address(info[4][0]))
    return addresses


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url(url: str, *, resolve: bool = True) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UrlBlocked("Only http and https URLs are permitted.")
    host = (parsed.hostname or "").rstrip(".")
    if not host:
        raise UrlBlocked("URL is missing a hostname.")
    if host in _BLOCKED_HOSTS or host.endswith(".local") or host.endswith(".internal"):
        raise UrlBlocked("Hostname is not allowed.")
    if resolve:
        for ip in _host_ips(host):
            if _is_disallowed_ip(ip):
                raise UrlBlocked(f"Resolved destination {ip} is not allowed.")
    return url


def validate_redirect_chain(start_url: str, hops: list[str]) -> None:
    validate_url(start_url)
    current = start_url
    for hop in hops:
        absolute = urljoin(current, hop)
        validate_url(absolute)
        current = absolute


def fetch_text(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    client: httpx.Client | None = None,
) -> tuple[str, str]:
    """Fetch readable text after validating the URL and every redirect hop."""
    validate_url(url)
    owns = client is None
    http = client or httpx.Client(follow_redirects=False, timeout=timeout)
    try:
        current = url
        seen: set[str] = set()
        for _ in range(5):
            if current in seen:
                raise UrlBlocked("Redirect loop detected.")
            seen.add(current)
            validate_url(current)
            response = http.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UrlBlocked("Redirect is missing a Location header.")
                current = urljoin(str(response.url), location)
                continue
            response.raise_for_status()
            validate_url(str(response.url))
            content = response.content[: max_bytes + 1]
            if len(content) > max_bytes:
                content = content[:max_bytes]
            text = content.decode(response.encoding or "utf-8", errors="replace")
            return str(response.url), text
        raise UrlBlocked("Too many redirects.")
    finally:
        if owns:
            http.close()
