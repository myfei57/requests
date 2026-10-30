"""
requests.sessions
~~~~~~~~~~~~~~~~~

This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from collections.abc import Generator, Iterator, Mapping, MutableMapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from ._internal_utils import to_native_string
from ._types import is_prepared as _is_prepared
from .adapters import HTTPAdapter
from .auth import _basic_auth_str
from .compat import cookielib, urljoin, urlparse
from .cookies import (
    RequestsCookieJar,
    cookiejar_from_dict,
    extract_cookies_to_jar,
    merge_cookies,
)
from .exceptions import (
    ChunkedEncodingError,
    ContentDecodingError,
    InvalidSchema,
    InvalidURL,
    PaginationCycleError,
    PaginationOriginError,
    TooManyRedirects,
)
from .hooks import default_hooks, dispatch_hook

# formerly defined here, reexposed here for backward compatibility
from .models import (  # noqa: F401
    DEFAULT_REDIRECT_LIMIT,
    REDIRECT_STATI,
    PreparedRequest,
    Request,
    Response,
)
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (  # noqa: F401
    DEFAULT_PORTS,
    default_headers,
    get_auth_from_url,
    get_environ_proxies,
    get_netrc_auth,
    requote_uri,
    resolve_proxies,
    rewind_body,
    should_bypass_proxies,  # re-export for external consumers
    to_key_val_list,
)

if TYPE_CHECKING:
    from http.cookiejar import CookieJar

    from typing_extensions import Self, Unpack

    from . import _types as _t
    from .adapters import BaseAdapter

# Preferred clock, based on which one is more accurate on a given system.
if sys.platform == "win32":
    preferred_clock = time.perf_counter
else:
    preferred_clock = time.time


def merge_setting(
    request_setting: Any, session_setting: Any, dict_class: type = OrderedDict
) -> Any:
    """Determines appropriate setting for a given request, taking into account
    the explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`
    """

    if session_setting is None:
        return request_setting

    if request_setting is None:
        return session_setting

    # Bypass if not a dictionary (e.g. verify)
    if not (
        isinstance(session_setting, Mapping) and isinstance(request_setting, Mapping)
    ):
        return request_setting

    merged_setting = dict_class(to_key_val_list(session_setting))  # type: ignore[arg-type]  # isinstance narrows Any to Mapping[Unknown]
    merged_setting.update(to_key_val_list(request_setting))  # type: ignore[arg-type]

    # Remove keys that are set to None. Extract keys first to avoid altering
    # the dictionary during iteration.
    none_keys = [k for (k, v) in merged_setting.items() if v is None]
    for key in none_keys:
        del merged_setting[key]

    return merged_setting


def merge_hooks(
    request_hooks: _t.HooksType,
    session_hooks: _t.HooksType,
    dict_class: type = OrderedDict,
) -> _t.HooksType:
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get("response") == []:
        return request_hooks

    if request_hooks is None or request_hooks.get("response") == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)


class SessionRedirectMixin:
    max_redirects: int
    trust_env: bool
    cookies: RequestsCookieJar

    def send(self, request: PreparedRequest, **kwargs: Any) -> Response: ...

    def get_redirect_target(self, resp: Response) -> str | None:
        """Receives a Response. Returns a redirect URI or ``None``"""
        # Due to the nature of how requests processes redirects this method will
        # be called at least once upon the original response and at least twice
        # on each subsequent redirect response (if any).
        # If a custom mixin is used to handle this logic, it may be advantageous
        # to cache the redirect location onto the response object as a private
        # attribute.
        if resp.is_redirect:
            location = resp.headers["location"]
            # Currently the underlying http module on py3 decode headers
            # in latin1, but empirical evidence suggests that latin1 is very
            # rarely used with non-ASCII characters in HTTP headers.
            # It is more likely to get UTF8 header rather than latin1.
            # This causes incorrect handling of UTF8 encoded location headers.
            # To solve this, we re-encode the location in latin1.
            location = location.encode("latin1")
            return to_native_string(location, "utf8")
        return None

    def should_strip_auth(self, old_url: str, new_url: str) -> bool:
        """Decide whether Authorization header should be removed when redirecting"""
        old_parsed = urlparse(old_url)
        new_parsed = urlparse(new_url)
        if old_parsed.hostname != new_parsed.hostname:
            return True
        # Special case: allow http -> https redirect when using the standard
        # ports. This isn't specified by RFC 7235, but is kept to avoid
        # breaking backwards compatibility with older versions of requests
        # that allowed any redirects on the same host.
        if (
            old_parsed.scheme == "http"
            and old_parsed.port in (80, None)
            and new_parsed.scheme == "https"
            and new_parsed.port in (443, None)
        ):
            return False

        # Handle default port usage corresponding to scheme.
        changed_port = old_parsed.port != new_parsed.port
        changed_scheme = old_parsed.scheme != new_parsed.scheme
        default_port = (DEFAULT_PORTS.get(old_parsed.scheme, None), None)
        if (
            not changed_scheme
            and old_parsed.port in default_port
            and new_parsed.port in default_port
        ):
            return False

        # Standard case: root URI must match
        return changed_port or changed_scheme

    def resolve_redirects(
        self,
        resp: Response,
        req: PreparedRequest,
        stream: bool = False,
        timeout: _t.TimeoutType = None,
        verify: _t.VerifyType = True,
        cert: _t.CertType = None,
        proxies: dict[str, str] | None = None,
        yield_requests: bool = False,
        **adapter_kwargs: Any,
    ) -> Generator[Response, None, None]:
        """Receives a Response. Returns a generator of Responses or Requests."""

        hist: list[Response] = []  # keep track of history

        url = self.get_redirect_target(resp)
        previous_fragment = urlparse(req.url).fragment
        while url:
            prepared_request = req.copy()

            # Update history and keep track of redirects.
            resp.history = hist[:]
            hist.append(resp)

            try:
                resp.content  # Consume socket so it can be released
            except (ChunkedEncodingError, ContentDecodingError, RuntimeError):
                resp.raw.read(decode_content=False)

            if len(resp.history) >= self.max_redirects:
                raise TooManyRedirects(
                    f"Exceeded {self.max_redirects} redirects.", response=resp
                )

            # Release the connection back into the pool.
            resp.close()

            # Handle redirection without scheme (see: RFC 1808 Section 4)
            if url.startswith("//"):
                parsed_rurl = urlparse(resp.url)
                url = ":".join([to_native_string(parsed_rurl.scheme), url])

            # Normalize url case and attach previous fragment if needed (RFC 7231 7.1.2)
            parsed = urlparse(url)
            if parsed.fragment == "" and previous_fragment:
                parsed = parsed._replace(fragment=previous_fragment)
            elif parsed.fragment:
                previous_fragment = parsed.fragment
            url = parsed.geturl()

            # Facilitate relative 'location' headers, as allowed by RFC 7231.
            # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
            # Compliant with RFC3986, we percent encode the url.
            if not parsed.netloc:
                url = urljoin(resp.url, requote_uri(url))
            else:
                url = requote_uri(url)

            prepared_request.url = to_native_string(url)

            self.rebuild_method(prepared_request, resp)

            # https://github.com/psf/requests/issues/1084
            if resp.status_code not in (
                codes.temporary_redirect,
                codes.permanent_redirect,
            ):
                # https://github.com/psf/requests/issues/3490
                purged_headers = ("Content-Length", "Content-Type", "Transfer-Encoding")
                for header in purged_headers:
                    prepared_request.headers.pop(header, None)
                prepared_request.body = None

            headers = prepared_request.headers
            headers.pop("Cookie", None)

            # Extract any cookies sent on the response to the cookiejar
            # in the new request. Because we've mutated our copied prepared
            # request, use the old one that we haven't yet touched.
            cookie_jar = cast("CookieJar", prepared_request._cookies)
            extract_cookies_to_jar(cookie_jar, req, resp.raw)
            merge_cookies(cookie_jar, self.cookies)
            prepared_request.prepare_cookies(cookie_jar)

            # Rebuild auth and proxy information.
            proxies = self.rebuild_proxies(prepared_request, proxies)
            self.rebuild_auth(prepared_request, resp)

            # A failed tell() sets `_body_position` to `object()`. This non-None
            # value ensures `rewindable` will be True, allowing us to raise an
            # UnrewindableBodyError, instead of hanging the connection.
            rewindable = prepared_request._body_position is not None and (
                "Content-Length" in headers or "Transfer-Encoding" in headers
            )

            # Attempt to rewind consumed file-like object.
            if rewindable:
                rewind_body(prepared_request)

            # Override the original request.
            req = prepared_request

            if yield_requests:
                yield req  # type: ignore[misc]  # Internal use only, returns PreparedRequest
            else:
                resp = self.send(
                    req,
                    stream=stream,
                    timeout=timeout,
                    verify=verify,
                    cert=cert,
                    proxies=proxies,
                    allow_redirects=False,
                    **adapter_kwargs,
                )

                extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)

                # extract redirect url, if any, for the next loop
                url = self.get_redirect_target(resp)
                yield resp

    def rebuild_auth(
        self, prepared_request: PreparedRequest, response: Response
    ) -> None:
        """When being redirected we may want to strip authentication from the
        request to avoid leaking credentials. This method intelligently removes
        and reapplies authentication where possible to avoid credential loss.
        """
        original_request = response.request
        assert _is_prepared(original_request)
        assert _is_prepared(prepared_request)

        headers = prepared_request.headers
        original_url = original_request.url
        url = prepared_request.url

        if "Authorization" in headers and self.should_strip_auth(original_url, url):
            # If we get redirected to a new host, we should strip out any
            # authentication headers.
            del headers["Authorization"]

        # .netrc might have more auth for us on our new host.
        new_auth = get_netrc_auth(url) if self.trust_env else None
        if new_auth is not None:
            prepared_request.prepare_auth(new_auth)

    def rebuild_proxies(
        self,
        prepared_request: PreparedRequest,
        proxies: dict[str, str] | None,
    ) -> dict[str, str]:
        """This method re-evaluates the proxy configuration by considering the
        environment variables. If we are redirected to a URL covered by
        NO_PROXY, we strip the proxy configuration. Otherwise, we set missing
        proxy keys for this URL (in case they were stripped by a previous
        redirect).

        This method also replaces the Proxy-Authorization header where
        necessary.

        :rtype: dict
        """
        assert _is_prepared(prepared_request)
        headers = prepared_request.headers
        scheme = urlparse(prepared_request.url).scheme
        new_proxies = resolve_proxies(prepared_request, proxies, self.trust_env)

        if "Proxy-Authorization" in headers:
            del headers["Proxy-Authorization"]

        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None

        # urllib3 handles proxy authorization for us in the standard adapter.
        # Avoid appending this to TLS tunneled requests where it may be leaked.
        if not scheme.startswith("https") and username and password:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return new_proxies

    def rebuild_method(
        self, prepared_request: PreparedRequest, response: Response
    ) -> None:
        """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
        method = prepared_request.method

        # https://tools.ietf.org/html/rfc7231#section-6.4.4
        if response.status_code == codes.see_other and method != "HEAD":
            method = "GET"

        # Do what the browsers do, despite standards...
        # First, turn 302s into GETs.
        if response.status_code == codes.found and method != "HEAD":
            method = "GET"

        # Second, if a POST is responded to with a 301, turn it into a GET.
        # This bizarre behaviour is explained in Issue 1704.
        if response.status_code == codes.moved and method == "POST":
            method = "GET"

        prepared_request.method = method


def _pagination_origin(url: str) -> tuple[str, str, int | None]:
    """Return the ``(scheme, lowercase host, effective port)`` of a URL.

    Explicit default ports are folded into their scheme's default port.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not parsed.scheme or hostname is None:
        raise InvalidURL(f"Invalid pagination URL: {url!r}")
    try:
        port = parsed.port
    except ValueError as e:
        raise InvalidURL(f"Invalid pagination URL: {url!r}") from e
    if port is None:
        port = DEFAULT_PORTS.get(parsed.scheme)
    return parsed.scheme, hostname.lower(), port


def _normalize_pagination_url(url: str) -> str:
    """Canonicalize a visited pagination URL.

    The fragment is ignored; the query string is preserved.
    """
    return urlparse(url)._replace(fragment="").geturl()


class LinkPaginationIterator(Iterator[Response]):
    """Lazily iterate GET responses connected by ``rel="next"`` Link headers.

    Returned by :meth:`Session.paginate`. No HTTP request is made until
    iteration starts, and the next page is only requested when the caller
    asks for it: pages are never prefetched.

    Each ``rel="next"`` target is resolved against the *final* URL of the
    current page (i.e. after redirects). The ``params`` passed to
    :meth:`~Session.paginate` apply to the first request only and are never
    re-appended to later pages; every page otherwise uses the session's
    authentication, cookies and connection configuration. Response bodies
    are always downloaded completely before the response is handed out.

    After iteration finishes, inspect :attr:`truncated` and :attr:`next_url`
    to tell a natural end (no ``rel="next"`` link) apart from a truncated one
    (the page limit was reached while another page was available).
    """

    def __init__(
        self,
        session: Session,
        url: _t.UriType,
        *,
        max_pages: int,
        params: _t.ParamsType = None,
        headers: _t.HeadersType = None,
        cookies: RequestsCookieJar | CookieJar | dict[str, str] | None = None,
        auth: _t.AuthType = None,
        timeout: _t.TimeoutType = None,
        proxies: dict[str, str] | None = None,
        hooks: _t.HooksInputType | None = None,
        verify: _t.VerifyType | None = None,
        cert: _t.CertType = None,
    ) -> None:
        # bool is a subclass of int; reject it explicitly.
        if isinstance(max_pages, bool) or not isinstance(max_pages, int):
            raise ValueError("max_pages must be a positive integer")
        if max_pages <= 0:
            raise ValueError("max_pages must be a positive integer")

        if isinstance(url, bytes):
            url = url.decode("utf-8")

        self._session = session
        self._url = url
        self._max_pages = max_pages
        self._params = params
        self._headers = headers
        self._cookies = cookies
        self._auth = auth
        self._timeout = timeout
        self._proxies = proxies
        self._hooks = hooks
        self._verify = verify
        self._cert = cert

        self._visited: set[str] = set()
        self._anchor: tuple[str, str, int | None] | None = None
        self._anchor_url: str | None = None
        self._pending_next: str | None = None
        self._undelivered: Response | None = None
        self._started = False
        self._finished = False
        self._closed = False

        #: Number of pages (delivered responses), the first page included.
        self.page_count: int = 0
        #: True when iteration stopped at ``max_pages`` with a next link left.
        self.truncated: bool = False
        #: Resolved ``rel="next"`` URL of the last delivered page, if any.
        self.next_url: str | None = None

    def __iter__(self) -> LinkPaginationIterator:
        return self

    def __next__(self) -> Response:
        if self._closed or self._finished:
            raise StopIteration

        response: Response | None = None
        try:
            if not self._started:
                self._started = True
                response = self._fetch_page(self._url, self._params, True)
                # The origin of the first page's *final* URL anchors every
                # later page and redirect hop.
                final_url = response.url
                self._anchor = _pagination_origin(final_url)
                self._anchor_url = final_url
            else:
                url = self._pending_next
                if url is None:
                    raise StopIteration
                # Both checks happen before the request is sent.
                self._assert_same_origin(url)
                if _normalize_pagination_url(url) in self._visited:
                    raise PaginationCycleError(
                        f"Pagination cycle detected: {url!r} was already visited."
                    )
                response = self._fetch_page(url, None, False)

                final_url = response.url
                if _normalize_pagination_url(final_url) in self._visited:
                    raise PaginationCycleError(
                        "Pagination cycle detected after redirects: "
                        f"{final_url!r} was already visited."
                    )

            self._undelivered = response
            self._visited.add(_normalize_pagination_url(response.url))
            self.page_count += 1

            next_url = self._extract_next_url(response)
            self.next_url = next_url

            if next_url is None:
                self._finished = True
            elif self.page_count >= self._max_pages:
                # Limit reached with another page available: stop cleanly and
                # report the truncation instead of fetching one more page.
                self.truncated = True
                self._finished = True
            else:
                self._pending_next = next_url

            self._undelivered = None
            return response
        except BaseException:
            if response is not None:
                self._release_response(response)
            self._undelivered = None
            self.close()
            raise

    def close(self) -> None:
        """Release any response fetched but not yet handed to the caller."""
        if self._undelivered is not None:
            self._release_response(self._undelivered)
            self._undelivered = None
        self._closed = True

    def __enter__(self) -> LinkPaginationIterator:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _assert_same_origin(self, url: str) -> None:
        assert self._anchor is not None
        if _pagination_origin(url) != self._anchor:
            raise PaginationOriginError(
                f"Refusing to follow pagination link to {url!r}: its scheme, "
                f"host or effective port differs from the first page "
                f"{self._anchor_url!r}."
            )

    def _extract_next_url(self, response: Response) -> str | None:
        link = response.links.get("next")
        if link is None:
            return None
        target = link.get("url")
        if not target:
            return None

        base_url = response.url
        # Protocol-relative link (see RFC 1808 Section 4), mirror the
        # redirect handling in SessionRedirectMixin.resolve_redirects.
        if target.startswith("//"):
            target = f"{urlparse(base_url).scheme}:{target}"

        parsed = urlparse(target)
        if not parsed.netloc:
            return urljoin(base_url, requote_uri(target))
        return requote_uri(target)

    def _prepare_page_request(self, url: str, params: _t.ParamsType) -> PreparedRequest:
        """Prepare a GET like :meth:`Session.prepare_request`, but with the
        exact (possibly empty) *params* given, without merging session params.
        """
        # Building the Request first normalizes the loose hooks input into
        # the dictionary shape expected by merge_hooks.
        page_request = Request(
            method="GET",
            url=url,
            headers=self._headers,
            cookies=self._cookies,
            auth=self._auth,
            hooks=self._hooks,
        )

        cookies = page_request.cookies or {}
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)
        merged_cookies = merge_cookies(
            merge_cookies(RequestsCookieJar(), self._session.cookies), cookies
        )

        auth = page_request.auth
        if self._session.trust_env and not auth and not self._session.auth:
            auth = get_netrc_auth(url)

        prepared = PreparedRequest()
        prepared.prepare(
            method=cast(str, page_request.method).upper(),
            url=cast("_t.UriType", page_request.url),
            files=page_request.files,
            data=page_request.data,
            json=page_request.json,
            headers=merge_setting(
                page_request.headers,
                self._session.headers,
                dict_class=CaseInsensitiveDict,
            ),
            params=params,
            auth=merge_setting(auth, self._session.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(page_request.hooks, self._session.hooks),
        )
        return prepared

    def _fetch_page(
        self, url: str, params: _t.ParamsType, merge_session_params: bool
    ) -> Response:
        # Session-level params belong to the first request only, just like
        # explicitly passed params: a rel="next" URL is already complete and
        # must not get them re-appended. Everything else (headers, auth,
        # cookies, hooks) still merges with the session configuration.
        if merge_session_params:
            params = merge_setting(params, self._session.params)
        prepared = self._prepare_page_request(url, params)
        assert _is_prepared(prepared)

        proxies = dict(self._proxies) if self._proxies else {}
        settings = self._session.merge_environment_settings(
            prepared.url, proxies, False, self._verify, self._cert
        )
        send_kwargs: dict[str, Any] = {"timeout": self._timeout}
        send_kwargs.update(settings)
        send_kwargs["allow_redirects"] = False
        # Pagination only supports the fully-buffered delivery mode.
        send_kwargs["stream"] = False

        response = self._session.send(prepared, **send_kwargs)
        history: list[Response] = []

        # Follow redirects manually so every hop can be checked against the
        # first page's origin *before* the request is sent.
        while response.is_redirect:
            if len(history) >= self._session.max_redirects:
                self._release_response(response)
                raise TooManyRedirects(
                    f"Exceeded {self._session.max_redirects} redirects.",
                    response=response,
                )

            redirects = self._session.resolve_redirects(
                response,
                prepared,
                stream=False,
                timeout=self._timeout,
                verify=send_kwargs["verify"],
                cert=send_kwargs["cert"],
                proxies=send_kwargs["proxies"],
                yield_requests=True,
            )
            try:
                next_request = cast("PreparedRequest", next(redirects))
            except StopIteration:
                break
            finally:
                redirects.close()

            next_url = cast(str, next_request.url)
            if self._anchor is not None:
                self._assert_same_origin(next_url)

            # resolve_redirects already consumed and closed this response.
            history.append(response)
            prepared = next_request
            send_kwargs["proxies"] = self._session.rebuild_proxies(
                prepared, send_kwargs["proxies"]
            )
            response = self._session.send(prepared, **send_kwargs)

        response.history = history
        # Guarantee the whole body has been read and the socket released.
        try:
            response.content
        except BaseException:
            # The response never reaches __next__ on failure, so release it
            # here instead of leaking the underlying connection.
            self._release_response(response)
            raise
        return response

    @staticmethod
    def _release_response(response: Response) -> None:
        try:
            response.content  # Consume socket so it can be released.
        except (ChunkedEncodingError, ContentDecodingError, RuntimeError):
            try:
                response.raw.read(decode_content=False)
            except Exception:
                pass
        response.close()


class Session(SessionRedirectMixin):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.get('https://httpbin.org/get')
      <Response [200]>

    Or as a context manager::

      >>> with requests.Session() as s:
      ...     s.get('https://httpbin.org/get')
      <Response [200]>
    """

    headers: CaseInsensitiveDict[str]
    auth: _t.AuthType
    proxies: dict[str, str]
    hooks: dict[str, list[_t.HookType]]
    params: MutableMapping[str, Any]
    stream: bool
    verify: _t.VerifyType
    cert: _t.CertType
    max_redirects: int
    trust_env: bool
    cookies: RequestsCookieJar
    adapters: MutableMapping[str, BaseAdapter]

    __attrs__: list[str] = [
        "headers",
        "cookies",
        "auth",
        "proxies",
        "hooks",
        "params",
        "verify",
        "cert",
        "adapters",
        "stream",
        "trust_env",
        "max_redirects",
    ]

    def __init__(self) -> None:
        #: A case-insensitive dictionary of headers to be sent on each
        #: :class:`Request <Request>` sent from this
        #: :class:`Session <Session>`.
        self.headers = default_headers()

        #: Default Authentication tuple or object to attach to
        #: :class:`Request <Request>`.
        self.auth = None

        #: Dictionary mapping protocol or protocol and host to the URL of the proxy
        #: (e.g. {'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}) to
        #: be used on each :class:`Request <Request>`.
        self.proxies = {}

        #: Event-handling hooks.
        self.hooks = default_hooks()

        #: Dictionary of querystring data to attach to each
        #: :class:`Request <Request>`. The dictionary values may be lists for
        #: representing multivalued query parameters.
        self.params = {}

        #: Stream response content default.
        self.stream = False

        #: SSL Verification default.
        #: Defaults to `True`, requiring requests to verify the TLS certificate at the
        #: remote end.
        #: If verify is set to `False`, requests will accept any TLS certificate
        #: presented by the server, and will ignore hostname mismatches and/or
        #: expired certificates, which will make your application vulnerable to
        #: man-in-the-middle (MitM) attacks.
        #: Only set this to `False` for testing.
        #: If verify is set to a string, it must be the path to a CA bundle file
        #: that will be used to verify the TLS certificate.
        self.verify = True

        #: SSL client certificate default, if String, path to ssl client
        #: cert file (.pem). If Tuple, ('cert', 'key') pair.
        self.cert = None

        #: Maximum number of redirects allowed. If the request exceeds this
        #: limit, a :class:`TooManyRedirects` exception is raised.
        #: This defaults to requests.models.DEFAULT_REDIRECT_LIMIT, which is
        #: 30.
        self.max_redirects = DEFAULT_REDIRECT_LIMIT

        #: Trust environment settings for proxy configuration, default
        #: authentication and similar.
        self.trust_env = True

        #: A CookieJar containing all currently outstanding cookies set on this
        #: session. By default it is a
        #: :class:`RequestsCookieJar <requests.cookies.RequestsCookieJar>`, but
        #: may be any other ``cookielib.CookieJar`` compatible object.
        self.cookies = cookiejar_from_dict({})

        # Default connection adapters.
        self.adapters = OrderedDict()
        self.mount("https://", HTTPAdapter())
        self.mount("http://", HTTPAdapter())

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def prepare_request(self, request: Request) -> PreparedRequest:
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for
        transmission and returns it. The :class:`PreparedRequest` has settings
        merged from the :class:`Request <Request>` instance and those of the
        :class:`Session`.

        :param request: :class:`Request` instance to prepare with this
            session's settings.
        :rtype: requests.PreparedRequest
        """
        url = cast("_t.UriType", request.url)
        method = cast(str, request.method)

        cookies = request.cookies or {}

        # Bootstrap CookieJar.
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)

        # Merge with session cookies
        merged_cookies = merge_cookies(
            merge_cookies(RequestsCookieJar(), self.cookies), cookies
        )

        # Set environment's basic authentication if not explicitly set.
        auth = request.auth
        if self.trust_env and not auth and not self.auth:
            auth = get_netrc_auth(url)

        p = PreparedRequest()
        p.prepare(
            method=method.upper(),
            url=url,
            files=request.files,
            data=request.data,
            json=request.json,
            headers=merge_setting(
                request.headers, self.headers, dict_class=CaseInsensitiveDict
            ),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p

    def request(
        self,
        method: str,
        url: _t.UriType,
        params: _t.ParamsType = None,
        data: _t.DataType = None,
        headers: _t.HeadersType = None,
        cookies: RequestsCookieJar | CookieJar | dict[str, str] | None = None,
        files: _t.FilesType = None,
        auth: _t.AuthType = None,
        timeout: _t.TimeoutType = None,
        allow_redirects: bool = True,
        proxies: dict[str, str] | None = None,
        hooks: _t.HooksInputType | None = None,
        stream: bool | None = None,
        verify: _t.VerifyType | None = None,
        cert: _t.CertType = None,
        json: _t.JsonType = None,
    ) -> Response:
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How many seconds to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param hooks: (optional) Dictionary mapping hook name to one event or
            list of events, event must be callable.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``. When set to
            ``False``, requests will accept any TLS certificate presented by
            the server, and will ignore hostname mismatches and/or expired
            certificates, which will make your application vulnerable to
            man-in-the-middle (MitM) attacks. Setting verify to ``False``
            may be useful during local development or testing.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """
        if isinstance(url, bytes):
            url = url.decode("utf-8")

        # Create the Request.
        req = Request(
            method=method.upper(),
            url=url,
            headers=headers,
            files=files,
            data=data or {},
            json=json,
            params=params or {},
            auth=auth,
            cookies=cookies,
            hooks=hooks,
        )
        prep = self.prepare_request(req)

        assert _is_prepared(prep)

        proxies = proxies or {}

        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )

        # Send the request.
        send_kwargs = {
            "timeout": timeout,
            "allow_redirects": allow_redirects,
        }
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)

        return resp

    def get(
        self,
        url: _t.UriType,
        params: _t.ParamsType = None,
        **kwargs: Unpack[_t.GetKwargs],
    ) -> Response:
        r"""Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary, list of tuples or bytes to send
            in the query string for the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("GET", url, params=params, **kwargs)

    def paginate(
        self,
        url: _t.UriType,
        *,
        max_pages: int,
        params: _t.ParamsType = None,
        headers: _t.HeadersType = None,
        cookies: RequestsCookieJar | CookieJar | dict[str, str] | None = None,
        auth: _t.AuthType = None,
        timeout: _t.TimeoutType = None,
        proxies: dict[str, str] | None = None,
        hooks: _t.HooksInputType | None = None,
        verify: _t.VerifyType | None = None,
        cert: _t.CertType = None,
    ) -> LinkPaginationIterator:
        r"""Iterate GET responses linked by ``rel="next"`` Link headers.

        Iteration starts with a GET to *url* and, for every returned page,
        follows its ``Link: <...>; rel="next"`` header (RFC 8288) until no
        next link is present. Pages are fetched lazily: a later page is only
        requested when the caller continues iterating, and it is never
        prefetched.

        A relative ``rel="next"`` target is resolved against the final URL of
        the current page (after redirects). *params* are sent with the first
        request only and are not re-appended to later pages; each page still
        uses this session's authentication, cookies and connection
        configuration. Response bodies are always fully read before delivery,
        so streaming is not supported.

        :param max_pages: Required positive integer upper bound on the number
            of pages returned; the first page counts towards it. If the bound
            is reached while another ``rel="next"`` link is present,
            iteration ends normally and the result is reported as truncated
            via the iterator's ``truncated`` and ``next_url`` attributes.
        :param params: (optional) Query parameters for the first request only.
        :param headers: (optional) Headers sent with every page request.
        :param cookies: (optional) Cookies sent with every page request.
        :param auth: (optional) Auth attached to every page request.
        :param timeout: (optional) Timeout applied to every page request.
        :param proxies: (optional) Proxies applied to every page request.
        :param hooks: (optional) Hooks registered on every page request.
        :param verify: (optional) TLS verification for every page request.
        :param cert: (optional) Client cert for every page request.
        :rtype: LinkPaginationIterator
        :raises PaginationCycleError: if a normalized URL (fragment ignored,
            query preserved) is visited more than once.
        :raises PaginationOriginError: if a next link or one of its
            redirects changes the scheme, host or effective port of the
            first page; the request is refused before being sent.
        """
        return LinkPaginationIterator(
            self,
            url,
            max_pages=max_pages,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            timeout=timeout,
            proxies=proxies,
            hooks=hooks,
            verify=verify,
            cert=cert,
        )

    def options(self, url: _t.UriType, **kwargs: Unpack[_t.RequestKwargs]) -> Response:
        r"""Sends a OPTIONS request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url: _t.UriType, **kwargs: Unpack[_t.RequestKwargs]) -> Response:
        r"""Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", False)
        return self.request("HEAD", url, **kwargs)

    def post(
        self,
        url: _t.UriType,
        data: _t.DataType = None,
        json: _t.JsonType = None,
        **kwargs: Unpack[_t.PostKwargs],
    ) -> Response:
        r"""Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("POST", url, data=data, json=json, **kwargs)

    def put(
        self, url: _t.UriType, data: _t.DataType = None, **kwargs: Unpack[_t.DataKwargs]
    ) -> Response:
        r"""Sends a PUT request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PUT", url, data=data, **kwargs)

    def patch(
        self, url: _t.UriType, data: _t.DataType = None, **kwargs: Unpack[_t.DataKwargs]
    ) -> Response:
        r"""Sends a PATCH request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PATCH", url, data=data, **kwargs)

    def delete(self, url: _t.UriType, **kwargs: Unpack[_t.RequestKwargs]) -> Response:
        r"""Sends a DELETE request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("DELETE", url, **kwargs)

    def send(self, request: PreparedRequest, **kwargs: Any) -> Response:
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        kwargs.setdefault("stream", self.stream)
        kwargs.setdefault("verify", self.verify)
        kwargs.setdefault("cert", self.cert)
        if "proxies" not in kwargs:
            kwargs["proxies"] = resolve_proxies(request, self.proxies, self.trust_env)

        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if isinstance(request, Request):
            raise ValueError("You can only send PreparedRequests.")

        assert _is_prepared(request)

        # Set up variables needed for resolve_redirects and dispatching of hooks
        allow_redirects = kwargs.pop("allow_redirects", True)
        stream = kwargs.get("stream")
        hooks = request.hooks

        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)

        # Start time (approximately) of the request
        start = preferred_clock()

        # Send the request
        r = adapter.send(request, **kwargs)

        # Total elapsed time of the request (approximately)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)

        # Response manipulation hooks
        r = dispatch_hook("response", hooks, r, **kwargs)

        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)

        extract_cookies_to_jar(self.cookies, request, r.raw)

        # Resolve redirects if allowed.
        if allow_redirects:
            # Redirect resolving generator.
            gen = self.resolve_redirects(r, request, **kwargs)
            history = [resp for resp in gen]
        else:
            history = []

        # Shuffle things around if there's history.
        if history:
            # Insert the first (original) request at the start
            history.insert(0, r)
            # Get the last request made
            r = history.pop()
            r.history = history

        # If redirects aren't being followed, store the response on the Request for Response.next().
        if not allow_redirects:
            try:
                r._next = next(  # type: ignore[assignment]  # yield_requests=True returns PreparedRequest
                    self.resolve_redirects(r, request, yield_requests=True, **kwargs)
                )
            except StopIteration:
                pass

        if not stream:
            r.content

        return r

    def merge_environment_settings(
        self,
        url: str,
        proxies: dict[str, str] | None,
        stream: bool | None,
        verify: _t.VerifyType | None,
        cert: _t.CertType,
    ) -> dict[str, Any]:
        """
        Check the environment and merge it with some settings.

        :rtype: dict
        """
        # Gather clues from the surrounding environment.
        if self.trust_env:
            # Set environment's proxies.
            no_proxy = proxies.get("no_proxy") if proxies is not None else None
            env_proxies = get_environ_proxies(url, no_proxy=no_proxy)
            if proxies is not None:
                for k, v in env_proxies.items():
                    proxies.setdefault(k, v)

            # Look for requests environment configuration
            # and be compatible with cURL.
            if verify is True or verify is None:
                verify = (
                    os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("CURL_CA_BUNDLE")
                    or verify
                )

        # Merge all the kwargs.
        proxies = merge_setting(proxies, self.proxies)
        stream = merge_setting(stream, self.stream)
        verify = merge_setting(verify, self.verify)
        cert = merge_setting(cert, self.cert)

        return {"proxies": proxies, "stream": stream, "verify": verify, "cert": cert}

    def get_adapter(self, url: str) -> BaseAdapter:
        """
        Returns the appropriate connection adapter for the given URL.

        :rtype: requests.adapters.BaseAdapter
        """
        for prefix, adapter in self.adapters.items():
            if url.lower().startswith(prefix.lower()):
                return adapter

        # Nothing matches :-/
        raise InvalidSchema(f"No connection adapters were found for {url!r}")

    def close(self) -> None:
        """Closes all adapters and as such the session"""
        for v in self.adapters.values():
            v.close()

    def mount(self, prefix: str, adapter: BaseAdapter) -> None:
        """Registers a connection adapter to a prefix.

        Adapters are sorted in descending order by prefix length.
        """
        self.adapters[prefix] = adapter
        keys_to_move = [k for k in self.adapters if len(k) < len(prefix)]

        for key in keys_to_move:
            self.adapters[key] = self.adapters.pop(key)

    def __getstate__(self) -> dict[str, Any]:
        state = {attr: getattr(self, attr, None) for attr in self.__attrs__}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        for attr, value in state.items():
            setattr(self, attr, value)


def session() -> Session:
    """
    Returns a :class:`Session` for context-management.

    .. deprecated:: 1.0.0

        This method has been deprecated since version 1.0.0 and is only kept for
        backwards compatibility. New code should use :class:`~requests.sessions.Session`
        to create a session. This may be removed at a future date.

    :rtype: Session
    """
    return Session()
