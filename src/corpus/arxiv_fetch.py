"""
The one place that talks to arXiv over HTTP, so a single clock governs every request.

In:  an `arxiv.Client` (whose session and delay clock we borrow) and a PDF URL.
Out: a downloaded file path, or `RateLimited` / `FetchError` raised for the caller
     to turn into an error dict. Never sleeps for longer than one request delay.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import arxiv
import requests


class FetchError(RuntimeError):
    """A PDF could not be retrieved. Ordinary, per-paper, keep going."""


class RateLimited(RuntimeError):
    """arXiv answered HTTP 429. Stop touching arXiv entirely, for minutes.

    Deliberately subclasses neither `arxiv.ArxivError` nor
    `requests.exceptions.ConnectionError`. `arxiv.Client._parse_feed` retries on
    exactly `(HTTPError, UnexpectedEmptyPageError, ConnectionError)`, so an exception
    outside that set propagates out of the search with zero retries. That is the point:
    once the IP is limited, arXiv answers 429 for *every* query, and the library's
    three retries are three more requests into the wall that deepen the block.
    """

    def __init__(self, url: str, retry_after: Optional[str] = None) -> None:
        self.url = url
        self.retry_after = retry_after
        suffix = f"; Retry-After: {retry_after}" if retry_after else ""
        super().__init__(
            f"arXiv returned HTTP 429 (rate limited) for {url}{suffix}. "
            "The limit is per-IP and applies to every arXiv request, not just this one."
        )


def _raise_on_429(response: "requests.Response", *args: Any, **kwargs: Any) -> None:
    """requests response hook. Raises out of `Session.send`, before any retry logic."""
    if response.status_code == 429:
        raise RateLimited(response.url, response.headers.get("Retry-After"))


def prepare_client(client: arxiv.Client, user_agent: str) -> arxiv.Client:
    """Attach the 429 hook and a real User-Agent to the client's session, once.

    Idempotent: `_build_client` reuses one client for the process, but calls through
    here on every search, and hooks would otherwise stack up.
    """
    session = client._session  # noqa: SLF001 — the library exposes no accessor
    hooks = session.hooks.setdefault("response", [])
    if _raise_on_429 not in hooks:
        hooks.append(_raise_on_429)
    session.headers["User-Agent"] = user_agent
    return client


def wait_for_slot(client: arxiv.Client) -> float:
    """Sleep until `client.delay_seconds` have passed since arXiv was last touched.

    Reads `arxiv.Client._last_request_dt`, the same private attribute the library's
    own feed path uses. Sharing it is the whole point: searches and PDF downloads hit
    the same host under the same per-IP budget, so they must share one clock. A
    separate clock for downloads would let a search and a download leave together.
    """
    last = getattr(client, "_last_request_dt", None)
    if last is None:
        return 0.0
    required = timedelta(seconds=client.delay_seconds)
    since = datetime.now() - last
    if since >= required:
        return 0.0
    to_sleep = (required - since).total_seconds()
    time.sleep(to_sleep)
    return to_sleep


def mark_request(client: arxiv.Client) -> None:
    """Record that arXiv was just touched, so the next caller waits for its slot."""
    client._last_request_dt = datetime.now()  # noqa: SLF001


def download_pdf(client: arxiv.Client, pdf_url: str, dest: Path, timeout_s: float) -> Path:
    """Fetch one PDF through the client's rate-limited session and atomically place it.

    Replaces `arxiv.Result.download_pdf`, which calls `urllib.request.urlretrieve`
    directly: no delay, no shared clock, no session, and a `Python-urllib` User-Agent.
    Eight results meant one politely spaced search followed by eight back-to-back PDF
    fetches — the burst that trips the per-IP limit.

    Writes to a sibling `.part` file and `os.replace`s it into position, so a crash or
    a 429 mid-body cannot leave a truncated PDF that the "already downloaded" check
    then treats as a cache hit forever.
    """
    if not pdf_url:
        raise FetchError("result carries no pdf_url")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    wait_for_slot(client)
    try:
        response = client._session.get(pdf_url, stream=True, timeout=timeout_s)  # noqa: SLF001
    except requests.RequestException as exc:
        raise FetchError(f"request failed: {exc}") from exc
    finally:
        # In the `finally` so it also stamps the failure paths: a request that errored
        # still left the machine and still counts against the per-IP budget.
        mark_request(client)

    try:
        if response.status_code != 200:
            raise FetchError(f"HTTP {response.status_code} for {pdf_url}")
        with open(part, "wb") as handle:
            for block in response.iter_content(chunk_size=65536):
                if block:
                    handle.write(block)
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        raise FetchError(f"download interrupted: {exc}") from exc
    except Exception:
        part.unlink(missing_ok=True)
        raise
    finally:
        response.close()

    # A 200 carrying an HTML error page is the failure a size check misses.
    with open(part, "rb") as handle:
        magic = handle.read(5)
    if not magic.startswith(b"%PDF"):
        part.unlink(missing_ok=True)
        raise FetchError(f"response body is not a PDF (starts {magic!r})")

    os.replace(part, dest)
    return dest


# --------------------------------------------------------------------------- cooldown
#
# A 429 blocks the IP for every arXiv request, not the one that tripped it. Nothing in
# the process remembers that, so the agent varying its wording, the eval harness moving
# to its next topic, or the user simply re-running all send fresh requests into the
# block and extend it. The memo is the smallest thing that stops that: an absolute
# expiry on disk, checked before any request, reported with the time it lifts.


def read_cooldown(path: Path) -> float:
    """Seconds still to wait before arXiv may be touched, or 0.0. Never raises.

    A corrupt or unreadable memo reads as 'no cooldown'. A memo that silently blocked
    collection would be worse than the burst it prevents.
    """
    try:
        memo = json.loads(Path(path).read_text(encoding="utf-8"))
        remaining = float(memo["until"]) - time.time()
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0
    return remaining if remaining > 0 else 0.0


def write_cooldown(path: Path, retry_after: Optional[str], default_s: float) -> float:
    """Record that arXiv just refused us. Returns the cooldown length in seconds.

    `Retry-After` is honoured when arXiv sends a parseable one; otherwise the configured
    default applies. Stored as an absolute expiry, not a duration, so a process that
    starts later does not restart the clock.
    """
    seconds = default_s
    if retry_after:
        try:
            seconds = max(float(retry_after), 0.0)
        except ValueError:
            seconds = default_s  # the HTTP-date form; not worth parsing for this
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"until": time.time() + seconds, "retry_after": retry_after, "seconds": seconds}
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # an unwritable cache must not turn a rate limit into a crash
    return seconds


def cooldown_detail(remaining: float, path: Path) -> str:
    """The message a caller shows for an active cooldown: when it lifts, how to clear it."""
    lifts = time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
    return (
        f"arXiv rate-limited this IP; not sending any request for another "
        f"{remaining:.0f}s (until about {lifts}). The limit is per-IP and applies to "
        f"every arXiv query, so retrying with different wording only extends it. "
        f"Delete {path} to clear this early."
    )
