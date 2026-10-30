"""Tests for ``Session.paginate`` (Link rel=next pagination)."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

import requests
from requests.exceptions import (
    PaginationCycleError,
    PaginationOriginError,
)
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict


class FakeRaw:
    """File-like stand-in for the subset of urllib3's HTTPResponse used."""

    def __init__(self, body: bytes = b"", read_error: Exception | None = None):
        self._body = body
        self._pos = 0
        self.closed = False
        self.released = False
        self._read_error = read_error

    def read(self, amount: int | None = None, **kwargs: object) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        if amount is None or amount < 0:
            chunk = self._body[self._pos :]
            self._pos = len(self._body)
        else:
            chunk = self._body[self._pos : self._pos + amount]
            self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True

    @property
    def fully_read(self) -> bool:
        return self._pos == len(self._body)


def make_response(
    url: str,
    *,
    status: int = 200,
    next: str | None = None,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
    read_error: Exception | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = url

    response_headers = CaseInsensitiveDict(headers or {})
    if next is not None:
        response_headers["Link"] = f'<{next}>; rel="next"'
    response.headers = response_headers

    response.raw = FakeRaw(body, read_error)

    prepared = PreparedRequest()
    prepared.prepare(method="GET", url=url)
    response.request = prepared
    return response


class FakeSession(requests.Session):
    """Real session with a scripted, network-free ``send``."""

    def __init__(self, script: list[object]):
        super().__init__()
        self.trust_env = False
        self._script = list(script)
        self.sent: list[PreparedRequest] = []
        self.send_kwargs: list[dict[str, object]] = []

    def send(self, request: PreparedRequest, **kwargs: object) -> requests.Response:
        self.sent.append(request)
        self.send_kwargs.append(kwargs)
        if not self._script:
            raise AssertionError(f"unexpected extra request to {request.url}")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    @property
    def sent_urls(self) -> list[str]:
        return [request.url for request in self.sent]


A = "http://api.test/a"
B = "http://api.test/b"
C = "http://api.test/c"


def chain(
    urls: list[str], *, bodies: bytes | list[bytes] = b""
) -> list[requests.Response]:
    if isinstance(bodies, bytes):
        bodies = [bodies] * len(urls)
    responses: list[requests.Response] = []
    for index, url in enumerate(urls):
        next_url = urls[index + 1] if index + 1 < len(urls) else None
        responses.append(make_response(url, next=next_url, body=bodies[index]))
    return responses


def test_paginate_walks_next_links_and_ends_naturally():
    session = FakeSession(chain([A, B, C], bodies=[b"one", b"two", b"three"]))

    pages = session.paginate(A, max_pages=10)
    assert pages.page_count == 0
    assert session.sent == []  # nothing is requested eagerly

    responses = list(pages)

    assert [response.url for response in responses] == [A, B, C]
    assert [response.content for response in responses] == [b"one", b"two", b"three"]
    assert session.sent_urls == [A, B, C]
    assert pages.page_count == 3
    assert pages.truncated is False
    assert pages.next_url is None
    # Delivered bodies were fully downloaded before being handed out.
    assert all(response.raw.fully_read for response in responses)


def test_paginate_is_lazy_between_pages():
    session = FakeSession(chain([A, B, C]))
    pages = session.paginate(A, max_pages=10)

    assert session.sent == []
    first = next(pages)
    assert session.sent_urls == [A]
    assert pages.next_url == B
    second = next(pages)
    assert session.sent_urls == [A, B]
    assert first is not second


def test_max_pages_is_required_and_validated():
    session = FakeSession(chain([A]))
    with pytest.raises(TypeError):
        session.paginate(A)  # type: ignore[call-arg]

    for invalid in (0, -3, 1.5, "2", True, False):
        with pytest.raises(ValueError):
            session.paginate(A, max_pages=invalid)  # type: ignore[arg-type]


def test_first_page_counts_towards_max_pages():
    session = FakeSession(chain([A, B, C]))
    pages = session.paginate(A, max_pages=1)

    responses = list(pages)
    assert [response.url for response in responses] == [A]
    assert pages.truncated is True
    assert pages.next_url == B
    assert session.sent_urls == [A]  # B was never requested

    with pytest.raises(StopIteration):
        next(pages)


def test_truncation_is_reported_explicitly_without_fetching_more():
    session = FakeSession(chain([A, B, C]))
    pages = session.paginate(A, max_pages=2)

    responses = list(pages)
    assert [response.url for response in responses] == [A, B]
    assert pages.page_count == 2
    assert pages.truncated is True
    assert pages.next_url == C
    assert session.sent_urls == [A, B]


def test_params_apply_to_first_page_only():
    session = FakeSession(
        [
            make_response("http://api.test/items", next="/items?page=2", body=b"one"),
            make_response(
                "http://api.test/items?page=2", next="/items?page=3", body=b"two"
            ),
            make_response("http://api.test/items?page=3", body=b"three"),
        ]
    )
    session.params = {"token": "x"}
    pages = session.paginate("http://api.test/items", params={"page": "1"}, max_pages=3)

    list(pages)

    first_query = parse_qs(urlsplit(session.sent_urls[0]).query)
    second_query = parse_qs(urlsplit(session.sent_urls[1]).query)
    third_query = parse_qs(urlsplit(session.sent_urls[2]).query)

    # First request carries both call params and session params.
    assert first_query == {"token": ["x"], "page": ["1"]}
    # Neither the first-page params nor session params are re-appended;
    # each rel="next" URL is requested verbatim.
    assert second_query == {"page": ["2"]}
    assert third_query == {"page": ["3"]}


def test_relative_link_resolved_against_final_url_after_redirect():
    session = FakeSession(
        [
            make_response(
                "http://api.test/old", status=302, headers={"Location": "/new"}
            ),
            make_response("http://api.test/new", next="page2?x=1"),
            make_response("http://api.test/page2?x=1"),
        ]
    )
    pages = session.paginate("http://api.test/old", max_pages=5)

    first = next(pages)
    assert first.url == "http://api.test/new"
    assert [response.status_code for response in first.history] == [302]
    assert first.history[0].headers["Location"] == "/new"

    second = next(pages)
    assert second.url == "http://api.test/page2?x=1"
    assert session.sent_urls == [
        "http://api.test/old",
        "http://api.test/new",
        "http://api.test/page2?x=1",
    ]
    assert list(pages) == []  # natural end


def test_cycle_on_link_is_reported_before_requesting():
    session = FakeSession(
        [
            make_response(A, next=B),
            make_response(B, next="http://api.test/a#section"),
        ]
    )
    pages = session.paginate(A, max_pages=10)

    assert next(pages).url == A
    assert next(pages).url == B
    with pytest.raises(PaginationCycleError) as excinfo:
        next(pages)
    assert "a#section" in str(excinfo.value)
    assert session.sent_urls == [A, B]  # the repeated URL was never sent

    with pytest.raises(StopIteration):
        next(pages)


def test_query_strings_are_significant_for_cycle_detection():
    base = "http://api.test/r"
    session = FakeSession(
        [
            make_response(f"{base}?n=1", next=f"{base}?n=2"),
            make_response(f"{base}?n=2", next=f"{base}?n=3"),
            make_response(f"{base}?n=3"),
        ]
    )
    pages = session.paginate(f"{base}?n=1", max_pages=5)

    assert pages.page_count == 0
    responses = list(pages)
    assert [response.url for response in responses] == [
        f"{base}?n=1",
        f"{base}?n=2",
        f"{base}?n=3",
    ]
    assert pages.truncated is False


def test_cycle_after_redirect_releases_internal_response():
    looping = make_response(
        "http://api.test/go", status=302, headers={"Location": "/a"}
    )
    final = make_response(A)
    session = FakeSession([make_response(A, next="/go"), looping, final])
    pages = session.paginate(A, max_pages=10)

    next(pages)
    with pytest.raises(PaginationCycleError):
        next(pages)

    # The redirect response and the final (duplicate) response were released.
    assert looping.raw.released is True
    assert final.raw.released is True
    assert session.sent_urls == [A, "http://api.test/go", A]


@pytest.mark.parametrize(
    "target",
    [
        "http://evil.test/b",
        "https://api.test/b",
        "http://api.test:8080/b",
        "//evil.test/b",
    ],
)
def test_cross_origin_link_rejected_before_request(target):
    session = FakeSession([make_response(A, next=target)])
    pages = session.paginate(A, max_pages=5)

    assert next(pages).url == A
    with pytest.raises(PaginationOriginError) as excinfo:
        next(pages)
    assert target in str(excinfo.value)
    assert session.sent_urls == [A]  # nothing was sent to the new origin


def test_explicit_default_port_matches_anchor():
    session = FakeSession(
        [make_response(A, next="http://api.test:80/b"), make_response(B)]
    )
    pages = session.paginate(A, max_pages=5)

    assert [response.url for response in pages] == [A, B]
    assert session.sent_urls == [A, "http://api.test:80/b"]


@pytest.mark.parametrize(
    "location",
    [
        "http://evil.test/c",
        "https://api.test/c",
        "http://api.test:8080/c",
    ],
)
def test_cross_origin_redirect_rejected_before_send(location):
    redirect = make_response(B, status=302, headers={"Location": location})
    session = FakeSession([make_response(A, next="/b"), redirect])
    pages = session.paginate(A, max_pages=5)

    next(pages)
    with pytest.raises(PaginationOriginError):
        next(pages)

    assert session.sent_urls == [A, B]  # the redirect target was never sent
    assert redirect.raw.released is True  # the redirect response was freed


def test_first_page_redirect_may_change_origin_and_reanchors():
    session = FakeSession(
        [
            make_response(
                "http://a.test/x",
                status=302,
                headers={"Location": "http://b.test/y"},
            ),
            make_response("http://b.test/y", next="/z"),
            make_response("http://b.test/z"),
        ]
    )
    pages = session.paginate("http://a.test/x", max_pages=5)

    responses = list(pages)
    assert responses[0].url == "http://b.test/y"
    assert responses[1].url == "http://b.test/z"
    assert session.sent_urls == [
        "http://a.test/x",
        "http://b.test/y",
        "http://b.test/z",
    ]


def test_session_auth_headers_and_cookies_used_on_every_page():
    session = FakeSession(chain([A, B]))
    session.auth = ("user", "pass")
    session.headers["X-Test"] = "42"
    session.cookies.set("cid", "xyz", domain="api.test", path="/")

    pages = session.paginate(A, max_pages=3)
    list(pages)

    assert len(session.sent) == 2
    for request in session.sent:
        assert request.headers["Authorization"] == "Basic dXNlcjpwYXNz"
        assert request.headers["X-Test"] == "42"
        assert "cid=xyz" in request.headers["Cookie"]


def test_every_page_uses_full_buffered_mode():
    session = FakeSession(chain([A, B]))
    list(session.paginate(A, max_pages=3))

    assert all(kwargs["stream"] is False for kwargs in session.send_kwargs)
    assert all(kwargs["allow_redirects"] is False for kwargs in session.send_kwargs)


def test_close_stops_iteration_without_prefetching():
    session = FakeSession(chain([A, B, C]))
    pages = session.paginate(A, max_pages=5)

    next(pages)
    pages.close()
    assert session.sent_urls == [A]

    with pytest.raises(StopIteration):
        next(pages)

    # Idempotent and usable as a context manager.
    pages.close()
    with session.paginate(A, max_pages=5) as pages2:
        next(pages2)
    assert session.sent_urls == [A, A]


def test_break_in_for_loop_does_not_prefetch():
    session = FakeSession(chain([A, B, C]))
    for response in session.paginate(A, max_pages=5):
        assert response.url == A
        break
    assert session.sent_urls == [A]


def test_send_failure_propagates_and_closes_iterator():
    session = FakeSession([requests.ConnectionError("boom"), make_response(A)])
    pages = session.paginate(A, max_pages=3)

    with pytest.raises(requests.ConnectionError):
        next(pages)

    with pytest.raises(StopIteration):
        next(pages)


def test_body_read_failure_releases_the_response():
    broken = make_response(A, read_error=RuntimeError("connection cut"))
    session = FakeSession([broken])
    pages = session.paginate(A, max_pages=3)

    with pytest.raises(RuntimeError):
        next(pages)

    assert broken.raw.closed is True
    with pytest.raises(StopIteration):
        next(pages)


def test_multiple_link_relations_picks_next():
    response = make_response(
        A,
        headers={
            "Link": '<http://api.test/b>; rel="next", <http://api.test/z>; rel="last"'
        },
    )
    session = FakeSession([response, make_response(B)])
    pages = session.paginate(A, max_pages=3)

    next(pages)
    assert pages.next_url == B
    assert next(pages).url == B


def test_bytes_start_url_is_accepted():
    session = FakeSession(chain([A, B]))
    pages = session.paginate(A.encode("utf-8"), max_pages=3)

    assert [response.url for response in pages] == [A, B]
