"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `http.cookiejar.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

from __future__ import annotations

import calendar
import copy
import time
from collections.abc import Iterator, MutableMapping
from http.cookiejar import Cookie, CookieJar, CookiePolicy, DefaultCookiePolicy
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ._internal_utils import to_native_string
from ._types import is_prepared as _is_prepared
from .compat import Morsel, cookielib, urlparse, urlunparse

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

    from .models import PreparedRequest

import threading


class MockRequest:
    """Wraps a `requests.PreparedRequest` to mimic a `urllib2.Request`.

    The code in `http.cookiejar.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """

    type: str

    def __init__(self, request: PreparedRequest) -> None:
        assert _is_prepared(request)
        self._r = request
        self._new_headers: dict[str, str] = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self) -> str:
        return self.type

    def get_host(self) -> str:
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self) -> str:
        return self.get_host()

    def get_full_url(self) -> str:
        # Only return the response's URL if the user hadn't set the Host
        # header
        if not self._r.headers.get("Host"):
            return self._r.url
        # If they did set it, retrieve it and reconstruct the expected domain
        host = to_native_string(self._r.headers["Host"], encoding="utf-8")
        parsed = urlparse(self._r.url)
        # Reconstruct the URL as we expect it
        return urlunparse(
            [
                parsed.scheme,
                host,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ]
        )

    def is_unverifiable(self) -> bool:
        return True

    def has_header(self, name: str) -> bool:
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name: str, default: str | None = None) -> str | None:
        return self._r.headers.get(name, self._new_headers.get(name, default))  # type: ignore[return-value]

    def add_header(self, key: str, val: str) -> None:
        """cookiejar has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError(
            "Cookie headers should be added with add_unredirected_header()"
        )

    def add_unredirected_header(self, name: str, value: str) -> None:
        self._new_headers[name] = value

    def get_new_headers(self) -> dict[str, str]:
        return self._new_headers

    @property
    def unverifiable(self) -> bool:
        return self.is_unverifiable()

    @property
    def origin_req_host(self) -> str:
        return self.get_origin_req_host()

    @property
    def host(self) -> str:
        return self.get_host()


class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `http.cookiejar` expects to see them.
    """

    def __init__(self, headers: Any) -> None:
        """Make a MockResponse for `cookiejar` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers

    def info(self) -> Any:
        return self._headers

    def getheaders(self, name: str) -> Any:
        self._headers.getheaders(name)


def extract_cookies_to_jar(
    jar: CookieJar, request: PreparedRequest, response: Any
) -> None:
    """Extract the cookies from the response into a CookieJar.

    :param jar: http.cookiejar.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    if not (hasattr(response, "_original_response") and response._original_response):
        return
    # the _original_response field is the wrapped httplib.HTTPResponse object,
    req = MockRequest(request)
    # pull out the HTTPMessage with the headers and put it in the mock:
    res = MockResponse(response._original_response.msg)
    jar.extract_cookies(res, req)  # type: ignore[arg-type]


def get_cookie_header(jar: CookieJar, request: PreparedRequest) -> str | None:
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
    r = MockRequest(request)
    jar.add_cookie_header(r)  # type: ignore[arg-type]
    return r.get_new_headers().get("Cookie")


class CookiePolicyExplainError(RuntimeError):
    """Cookie send-selection cannot be safely explained.

    Raised by :func:`explain_cookie_send` when the jar's
    :class:`~http.cookiejar.CookiePolicy` (or the jar itself) overrides the
    standard :class:`~http.cookiejar.DefaultCookiePolicy` decision methods.
    In that case the per-cookie reasons reported here could not be trusted to
    match the cookies actually sent, so no explanation is fabricated.

    The error message intentionally contains neither cookie names/values, nor
    any request header or body content -- only the offending class name.
    """


class CookieSendChoice:
    """Explanation of whether one cookie in the jar would be sent.

    Cookies sharing a name but having a different ``domain`` or ``path`` are
    always reported as separate :class:`CookieSendChoice` objects.

    Cookie *values* are deliberately not part of this object.

    .. attribute:: name
    .. attribute:: domain
    .. attribute:: path
    .. attribute:: host_only
        ``True`` when the cookie was accepted without a ``Domain`` attribute.
    .. attribute:: secure
        ``True`` when the cookie is only sent over secure protocols.
    .. attribute:: expired
        ``True`` when the cookie is expired at the single point in time at
        which the whole explanation was evaluated.
    .. attribute:: selected
        ``True`` exactly when this cookie would appear in the ``Cookie``
        header produced for the same jar and :class:`PreparedRequest`.
    .. attribute:: reason
        Human-readable reason for the decision. Never contains cookie values.
    """

    __slots__ = (
        "name",
        "domain",
        "path",
        "host_only",
        "secure",
        "expired",
        "selected",
        "reason",
    )

    def __init__(
        self,
        name: str,
        domain: str,
        path: str,
        host_only: bool,
        secure: bool,
        expired: bool,
        selected: bool,
        reason: str,
    ) -> None:
        self.name = name
        self.domain = domain
        self.path = path
        self.host_only = host_only
        self.secure = secure
        self.expired = expired
        self.selected = selected
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"CookieSendChoice(name={self.name!r}, domain={self.domain!r}, "
            f"path={self.path!r}, host_only={self.host_only!r}, "
            f"secure={self.secure!r}, expired={self.expired!r}, "
            f"selected={self.selected!r})"
        )


class CookieSendExplanation:
    """Result of :func:`explain_cookie_send`.

    .. attribute:: url_host
        Host taken from the request URL.
    .. attribute:: host
        Effective host used for domain matching. Identical to ``url_host``
        unless the request overrides it with an explicit ``Host`` header.
    .. attribute:: host_overridden
        ``True`` when the request carries a ``Host`` header that replaces the
        URL host for cookie matching.
    .. attribute:: scheme
        Request scheme (``"http"``/``"https"``) used for the secure check.
    .. attribute:: path
        Request path used for path matching (never includes the query string).
    .. attribute:: explicit_cookie_header
        ``True`` when the request already has a ``Cookie`` header. When so,
        that header is sent as-is and takes precedence; no jar cookies are
        added, and the header's content is never read or displayed.
    .. attribute:: note
        Short summary of the overall decision.
    .. attribute:: cookies
        One :class:`CookieSendChoice` per cookie in the jar, in jar storage
        order. Same-named cookies with different domain/path are listed
        separately.
    """

    __slots__ = (
        "url_host",
        "host",
        "host_overridden",
        "scheme",
        "path",
        "explicit_cookie_header",
        "note",
        "cookies",
    )

    def __init__(
        self,
        *,
        url_host: str,
        host: str,
        host_overridden: bool,
        scheme: str,
        path: str,
        explicit_cookie_header: bool,
        note: str,
        cookies: list[CookieSendChoice],
    ) -> None:
        self.url_host = url_host
        self.host = host
        self.host_overridden = host_overridden
        self.scheme = scheme
        self.path = path
        self.explicit_cookie_header = explicit_cookie_header
        self.note = note
        self.cookies = cookies

    def selected_cookies(self) -> list[CookieSendChoice]:
        """Return only the choices that would actually be sent."""
        return [choice for choice in self.cookies if choice.selected]

    def __repr__(self) -> str:
        selected = sum(1 for choice in self.cookies if choice.selected)
        return (
            f"CookieSendExplanation(host={self.host!r}, scheme={self.scheme!r}, "
            f"path={self.path!r}, explicit_cookie_header="
            f"{self.explicit_cookie_header!r}, selected={selected}, "
            f"candidates={len(self.cookies)})"
        )


# Selection-relevant DefaultCookiePolicy methods. If any of these is
# overridden by a custom policy, standard reasons cannot be guaranteed to
# describe that policy's actual decisions.
_EXPLAIN_POLICY_METHODS: tuple[str, ...] = (
    "return_ok",
    "return_ok_version",
    "return_ok_verifiability",
    "return_ok_secure",
    "return_ok_expires",
    "return_ok_port",
    "return_ok_domain",
    "domain_return_ok",
    "path_return_ok",
    "is_blocked",
    "is_not_allowed",
)

# Selection-relevant CookieJar methods. Custom jars (or future stdlib
# versions) overriding these select cookies differently than explained here.
_EXPLAIN_JAR_METHODS: tuple[str, ...] = (
    "add_cookie_header",
    "_cookies_for_request",
    "_cookies_for_domain",
)

_EXPLICIT_HEADER_REASON = (
    "the request already has an explicit Cookie header, which is sent "
    "unchanged and takes precedence; no cookies from the jar are added to it"
)


# Thin, typed wrappers around untyped http.cookiejar helpers used by the
# explanation. They only read request metadata and never mutate anything.
def _request_path(req: MockRequest) -> str:
    return cast(str, cookielib.request_path(req))  # type: ignore[attr-defined]


def _request_port(req: MockRequest) -> str | None:
    return cast("str | None", cookielib.request_port(req))  # type: ignore[attr-defined]


def _safe_request_host(req: MockRequest) -> str:
    """Effective request host safe for display.

    Uses ``urlparse().hostname`` so userinfo (``user:password@``) embedded in
    a URL and port numbers can never appear in explanation output, while
    still honouring the ``Host``-header override applied by
    :meth:`MockRequest.get_full_url`.
    """
    return urlparse(req.get_full_url()).hostname or ""


def _explain_supported(jar: CookieJar, policy: CookiePolicy) -> None:
    """Raise unless jar/policy use the unmodified standard selection rules."""
    if not isinstance(policy, DefaultCookiePolicy):
        raise CookiePolicyExplainError(
            "Cannot safely explain cookie send-selection: the jar uses "
            f"cookie policy {type(policy).__name__!r}, which is not the "
            "standard http.cookiejar.DefaultCookiePolicy."
        )

    for method_name in _EXPLAIN_POLICY_METHODS:
        method = getattr(type(policy), method_name, None)
        standard = getattr(DefaultCookiePolicy, method_name, None)
        if method is None or standard is None or method is not standard:
            raise CookiePolicyExplainError(
                "Cannot safely explain cookie send-selection: the jar's "
                f"policy {type(policy).__name__!r} overrides the standard "
                f"decision method {method_name}(), so per-cookie reasons "
                "could not be guaranteed to match the cookies actually sent."
            )

    for method_name in _EXPLAIN_JAR_METHODS:
        method = getattr(type(jar), method_name, None)
        standard = getattr(CookieJar, method_name, None)
        if method is None or standard is None or method is not standard:
            raise CookiePolicyExplainError(
                "Cannot safely explain cookie send-selection: cookie jar "
                f"{type(jar).__name__!r} overrides the standard selection "
                f"method {method_name}(), so per-cookie reasons could not be "
                "guaranteed to match the cookies actually sent."
            )

    if not isinstance(getattr(jar, "_cookies", None), dict):
        raise CookiePolicyExplainError(
            "Cannot safely explain cookie send-selection: cookie jar "
            f"{type(jar).__name__!r} does not use the standard cookie "
            "storage layout."
        )


def _domain_gate_reason(
    policy: DefaultCookiePolicy, domain: str, req: MockRequest
) -> str:
    """Classify *why* a cookie-domain key failed domain_return_ok().

    The pass/fail decision itself comes from the policy; this only produces
    the human-readable label, so it never affects selection.
    """
    if policy.is_blocked(domain):
        return (
            f"cookie domain {domain!r} is on the policy block-list, so it is "
            "not returned to this host"
        )
    if policy.is_not_allowed(domain):
        return (
            f"cookie domain {domain!r} is not on the policy allow-list, so it "
            "is not returned to this host"
        )
    effective_host = _safe_request_host(req)
    return (
        f"cookie domain {domain!r} does not domain-match the effective "
        f"request host {effective_host!r}; cookies are only returned to the "
        "same host or one of its subdomains"
    )


def _cookie_return_reason(
    policy: DefaultCookiePolicy,
    cookie: Cookie,
    req: MockRequest,
    now: int,
) -> tuple[bool, str | None]:
    """Mirror DefaultCookiePolicy.return_ok() at a fixed point in time.

    Each predicate is the standard library's own method, called in the same
    order as ``DefaultCookiePolicy.return_ok()``. Expiry is checked with the
    caller-supplied timestamp instead of ``policy._now`` so that nothing on
    the jar or policy is mutated.
    """
    if not policy.return_ok_version(cookie, req):  # type: ignore[arg-type]
        if (cookie.version or 0) > 0:
            return False, (
                f"RFC 2965 (version {cookie.version}) cookies are switched "
                "off in the jar's policy"
            )
        return False, "Netscape cookies are switched off in the jar's policy"

    if not policy.return_ok_verifiability(cookie, req):  # type: ignore[arg-type]
        return False, (
            "third-party cookie is not returned for this unverifiable "
            "request by the jar's policy"
        )

    if not policy.return_ok_secure(cookie, req):  # type: ignore[arg-type]
        protocols = cast(
            "tuple[str, ...]",
            getattr(policy, "secure_protocols", ("https", "wss")),
        )
        return False, (
            f"secure cookie is not sent over the {req.type!r} scheme; it is "
            f"only sent over one of: {', '.join(sorted(protocols))}"
        )

    # Mirrors DefaultCookiePolicy.return_ok_expires() with policy._now == now.
    if cookie.is_expired(now):
        return False, (
            f"cookie has expired (expires={cookie.expires}, evaluated at {now})"
        )

    if not policy.return_ok_port(cookie, req):  # type: ignore[arg-type]
        req_port = _request_port(req)
        return False, (
            f"request port {req_port!r} does not match cookie port {cookie.port!r}"
        )

    if not policy.return_ok_domain(cookie, req):  # type: ignore[arg-type]
        effective_host = _safe_request_host(req)
        kind = "host-only" if not cookie.domain_specified else "domain"
        return False, (
            f"the {kind} cookie domain {cookie.domain!r} does not "
            f"domain-match the effective request host {effective_host!r}"
        )

    return True, (
        "cookie matches the request host, path and policy; it would be sent "
        "in the Cookie header"
    )


def _stdlib_selected_cookies(
    policy: DefaultCookiePolicy,
    storage: dict[str, dict[str, dict[str, Cookie]]],
    req: MockRequest,
    now: int,
) -> set[tuple[str, str, str]]:
    """Return the stdlib-selected cookies without touching the caller's jar.

    A throwaway jar is built from copies of the policy and cookie storage, so
    the standard library's own ``_cookies_for_request()`` decides at the same
    point in time while the user's jar, its policy, cookies and request are
    never mutated (in particular, ``add_cookie_header()`` and its
    ``clear_expired_cookies()`` call are never invoked on the real jar).
    """
    policy_copy = copy.copy(policy)
    setattr(policy_copy, "_now", now)
    probe = CookieJar(policy=policy_copy)
    setattr(probe, "_cookies", copy.deepcopy(storage))
    selected = probe._cookies_for_request(req)  # type: ignore[attr-defined,arg-type]
    return {
        (cookie.name, cookie.domain, cookie.path)
        for cookie in cast("list[Cookie]", selected)
    }


def explain_cookie_send(
    jar: CookieJar, request: PreparedRequest
) -> CookieSendExplanation:
    """Explain which cookies from *jar* would be sent with *request*.

    This is a local, read-only debugging aid: it does not send the request
    and does not modify the jar, its policy, any cookie, the request or any
    response. It evaluates every cookie against the exact same rules the
    standard library uses when building the real ``Cookie`` header
    (host/domain matching including subdomains and host-only cookies, path
    matching, the secure/scheme check, expiry and ``Host``-header override).

    Expiry is evaluated once, at a single point in time shared by every
    cookie. The set of cookies reported as ``selected`` is verified against
    the standard library's own selection for the same jar and
    :class:`PreparedRequest`, so it matches the cookies that would appear in
    the actual ``Cookie`` header.

    Cookies sharing a name but differing in ``domain`` or ``path`` are
    reported as separate entries; results are never keyed by name.

    When the request already carries a ``Cookie`` header, that header takes
    precedence: no jar cookies are reported as selected. The existing
    header's content is never read or displayed.

    No cookie value, authentication material, request header value or
    request body is ever included in the result or in raised errors.

    :param jar: any standard ``http.cookiejar.CookieJar`` (including
        :class:`RequestsCookieJar`, ``LWPCookieJar`` and ``MozillaCookieJar``)
        whose policy is an unmodified
        :class:`~http.cookiejar.DefaultCookiePolicy`.
    :param request: a prepared :class:`PreparedRequest`.
    :raises TypeError: if *jar* is not a ``CookieJar``.
    :raises CookiePolicyExplainError: if the jar or its
        :class:`~http.cookiejar.CookiePolicy` overrides the standard
        selection rules, so that safe per-cookie reasons cannot be given.
    """
    if not isinstance(jar, CookieJar):
        raise TypeError(
            f"explain_cookie_send() expected a CookieJar, got "
            f"{type(jar).__name__!r} instead"
        )
    assert _is_prepared(request)

    policy = cast(CookiePolicy, getattr(jar, "_policy"))
    _explain_supported(jar, policy)
    default_policy = cast(DefaultCookiePolicy, policy)

    # Single timestamp is used for every expiry decision in this call.
    now = int(time.time())

    req = MockRequest(request)

    parsed = urlparse(request.url)
    url_host = parsed.hostname or ""
    # Effective host, honouring a Host-header override, sanitised so that
    # URL userinfo can never appear in the explanation.
    req_host = _safe_request_host(req)
    host_overridden = bool(req.get_header("Host")) and req_host != url_host
    req_path = _request_path(req)

    # Membership only -- the existing header value is never read.
    explicit_cookie_header = req.has_header("Cookie")

    choices: list[CookieSendChoice] = []
    storage = cast("dict[str, dict[str, dict[str, Cookie]]]", getattr(jar, "_cookies"))

    for domain in list(storage.keys()):
        cookies_by_path = storage[domain]
        # Standard CookieJar domain-key optimisation gate.
        domain_ok = default_policy.domain_return_ok(domain, req)  # type: ignore[arg-type]
        domain_reason: str | None = None
        if not domain_ok:
            domain_reason = _domain_gate_reason(default_policy, domain, req)

        for path in list(cookies_by_path.keys()):
            cookies_by_name = cookies_by_path[path]
            path_ok = domain_ok and default_policy.path_return_ok(path, req)  # type: ignore[arg-type]

            for name in list(cookies_by_name.keys()):
                cookie = cookies_by_name[name]

                expired = bool(cookie.is_expired(now))
                host_only = not cookie.domain_specified

                if not domain_ok:
                    would_select = False
                    reason: str | None = domain_reason
                elif not path_ok:
                    would_select = False
                    reason = (
                        f"request path {req_path!r} does not path-match "
                        f"cookie path {path!r}; the request path must equal "
                        "the cookie path or be a sub-path of it"
                    )
                else:
                    would_select, reason = _cookie_return_reason(
                        default_policy, cookie, req, now
                    )

                selected = would_select and not explicit_cookie_header
                if would_select and explicit_cookie_header:
                    reason = _EXPLICIT_HEADER_REASON

                choices.append(
                    CookieSendChoice(
                        name=cookie.name,
                        domain=cookie.domain,
                        path=cookie.path,
                        host_only=host_only,
                        secure=bool(cookie.secure),
                        expired=expired,
                        selected=selected,
                        reason=cast(str, reason),
                    )
                )

    # Verify our replicated decisions against the standard library's own
    # selection on an isolated copy, evaluated at the exact same timestamp.
    explained_selection = {
        (choice.name, choice.domain, choice.path)
        for choice in choices
        if choice.selected
        or (choice.reason == _EXPLICIT_HEADER_REASON and not choice.selected)
    }
    actual_selection = _stdlib_selected_cookies(default_policy, storage, req, now)
    if explained_selection != actual_selection:
        raise CookiePolicyExplainError(
            "Cannot safely explain cookie send-selection: the explained "
            "selection does not match the cookies selected by the standard "
            "library for this jar and request."
        )

    if explicit_cookie_header:
        note = (
            "The request already carries a Cookie header, which takes "
            "precedence and is sent unchanged: no cookies from the jar are "
            "added, and the existing header was neither read nor displayed."
        )
    else:
        selected_count = len(actual_selection)
        note = (
            f"{selected_count} of {len(choices)} cookie(s) in the jar would "
            "be sent in the Cookie header for this request."
        )
        if host_overridden:
            note += (
                f" Cookie matching uses the explicit Host header host "
                f"{req_host!r} rather than the URL host {url_host!r}."
            )

    return CookieSendExplanation(
        url_host=url_host,
        host=req_host,
        host_overridden=host_overridden,
        scheme=req.type,
        path=req_path,
        explicit_cookie_header=explicit_cookie_header,
        note=note,
        cookies=choices,
    )


def remove_cookie_by_name(
    cookiejar: CookieJar, name: str, domain: str | None = None, path: str | None = None
) -> None:
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    clearables: list[tuple[str, str, str]] = []
    for cookie in cookiejar:
        if cookie.name != name:
            continue
        if domain is not None and domain != cookie.domain:
            continue
        if path is not None and path != cookie.path:
            continue
        clearables.append((cookie.domain, cookie.path, cookie.name))

    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """


class RequestsCookieJar(CookieJar, MutableMapping[str, str | None]):  # type: ignore[misc]
    """Compatibility class; is a http.cookiejar.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """

    _policy: CookiePolicy

    def get(  # type: ignore[override]
        self,
        name: str,
        default: str | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> str | None:
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default

    def set(
        self, name: str, value: str | Morsel[dict[str, str]] | None, **kwargs: Any
    ) -> Cookie | None:
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
        # support client code that unsets cookies by assignment of a None value:
        if value is None:
            remove_cookie_by_name(
                self, name, domain=kwargs.get("domain"), path=kwargs.get("path")
            )
            return

        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c

    def iterkeys(self) -> Iterator[str]:
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.name

    def keys(self) -> list[str]:  # type: ignore[override]
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
        return list(self.iterkeys())

    def itervalues(self) -> Iterator[str | None]:
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.value

    def values(self) -> list[str | None]:  # type: ignore[override]
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
        return list(self.itervalues())

    def iteritems(self) -> Iterator[tuple[str, str | None]]:
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
        for cookie in iter(self):
            yield cookie.name, cookie.value

    def items(self) -> list[tuple[str, str | None]]:  # type: ignore[override]
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
        return list(self.iteritems())

    def list_domains(self) -> list[str]:
        """Utility method to list all the domains in the jar."""
        domains: list[str] = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self) -> list[str]:
        """Utility method to list all the paths in the jar."""
        paths: list[str] = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self) -> bool:
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
        domains: list[str] = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:  # type: ignore[reportUnnecessaryComparison]  # defensive check
                return True
            domains.append(cookie.domain)
        return False  # there is only one domain in jar

    def get_dict(
        self, domain: str | None = None, path: str | None = None
    ) -> dict[str, str | None]:
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
        dictionary: dict[str, str | None] = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (
                path is None or cookie.path == path
            ):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __iter__(self) -> Iterator[Cookie]:  # type: ignore[override]
        """RequestCookieJar's __iter__ comes from CookieJar not MutableMapping."""
        return super().__iter__()

    def __contains__(self, name: object) -> bool:
        try:
            return super().__contains__(name)
        except CookieConflictError:
            return True

    def __getitem__(self, name: str) -> str | None:
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

        .. warning:: operation is O(n), not O(1).
        """
        return self._find_no_duplicates(name)

    def __setitem__(
        self, name: str, value: str | Morsel[dict[str, str]] | None
    ) -> None:
        """Dict-like __setitem__ for compatibility with client code. Throws
        exception if there is already a cookie of that name in the jar. In that
        case, use the more explicit set() method instead.
        """
        self.set(name, value)

    def __delitem__(self, name: str) -> None:
        """Deletes a cookie given a name. Wraps ``http.cookiejar.CookieJar``'s
        ``remove_cookie_by_name()``.
        """
        remove_cookie_by_name(self, name)

    def set_cookie(self, cookie: Cookie, *args: Any, **kwargs: Any) -> None:
        if (
            (value := cookie.value) is not None
            and value.startswith('"')
            and value.endswith('"')
        ):
            cookie.value = value.replace('\\"', "")
        return super().set_cookie(cookie, *args, **kwargs)

    def update(  # type: ignore[override]
        self, other: CookieJar | SupportsKeysAndGetItem[str, str]
    ) -> None:
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(copy.copy(cookie))
        else:
            super().update(other)

    def _find(
        self, name: str, domain: str | None = None, path: str | None = None
    ) -> str | None:
        """Requests uses this method internally to get cookie values.

        If there are conflicting cookies, _find arbitrarily chooses one.
        See _find_no_duplicates if you want an exception thrown if there are
        conflicting cookies.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :return: cookie.value
        """
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        return cookie.value

        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def _find_no_duplicates(
        self, name: str, domain: str | None = None, path: str | None = None
    ) -> str:
        """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if toReturn is not None:
                            # if there are multiple cookies that meet passed in criteria
                            raise CookieConflictError(
                                f"There are multiple cookies with name, {name!r}"
                            )
                        # we will eventually return this as long as no cookie conflict
                        toReturn = cookie.value

        if toReturn is not None:
            return toReturn
        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def __getstate__(self) -> dict[str, Any]:
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        # remove the unpickleable RLock object
        state.pop("_cookies_lock")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if "_cookies_lock" not in self.__dict__:
            self._cookies_lock = threading.RLock()

    def copy(self) -> RequestsCookieJar:
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar()
        new_cj.set_policy(self.get_policy())
        new_cj.update(self)
        return new_cj

    def get_policy(self) -> CookiePolicy:
        """Return the CookiePolicy instance used."""
        return self._policy


def _copy_cookie_jar(jar: CookieJar | None) -> CookieJar | None:  # type: ignore[reportUnusedFunction]  # cross-module usage in models.py
    if jar is None:
        return None

    if copy_method := getattr(jar, "copy", None):
        # We're dealing with an instance of RequestsCookieJar
        return copy_method()
    # We're dealing with a generic CookieJar instance
    new_jar = copy.copy(jar)
    new_jar.clear()
    for cookie in jar:
        new_jar.set_cookie(copy.copy(cookie))
    return new_jar


def create_cookie(name: str, value: str, **kwargs: Any) -> Cookie:
    """Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain ''
    and sent on every request (this is sometimes called a "supercookie").
    """
    result: dict[str, Any] = {
        "version": 0,
        "name": name,
        "value": value,
        "port": None,
        "domain": "",
        "path": "/",
        "secure": False,
        "expires": None,
        "discard": True,
        "comment": None,
        "comment_url": None,
        "rest": {"HttpOnly": None},
        "rfc2109": False,
    }

    badargs = set(kwargs) - set(result)
    if badargs:
        raise TypeError(
            f"create_cookie() got unexpected keyword arguments: {list(badargs)}"
        )

    result.update(kwargs)
    result["port_specified"] = bool(result["port"])
    result["domain_specified"] = bool(result["domain"])
    result["domain_initial_dot"] = result["domain"].startswith(".")
    result["path_specified"] = bool(result["path"])

    return cookielib.Cookie(**result)


def morsel_to_cookie(morsel: Morsel[Any]) -> Cookie:
    """Convert a Morsel object into a Cookie containing the one k/v pair."""

    expires: int | None = None
    if morsel["max-age"]:
        try:
            expires = int(time.time() + int(morsel["max-age"]))
        except ValueError:
            raise TypeError(f"max-age: {morsel['max-age']} must be integer")
    elif morsel["expires"]:
        time_template = "%a, %d-%b-%Y %H:%M:%S GMT"
        expires = calendar.timegm(time.strptime(morsel["expires"], time_template))
    return create_cookie(
        comment=morsel["comment"],
        comment_url=bool(morsel["comment"]),
        discard=False,
        domain=morsel["domain"],
        expires=expires,
        name=morsel.key,
        path=morsel["path"],
        port=None,
        rest={"HttpOnly": morsel["httponly"]},
        rfc2109=False,
        secure=bool(morsel["secure"]),
        value=morsel.value,
        version=morsel["version"] or 0,
    )


_CookieJarT = TypeVar("_CookieJarT", bound=CookieJar)


@overload
def cookiejar_from_dict(
    cookie_dict: dict[str, str] | None,
    cookiejar: None = None,
    overwrite: bool = True,
) -> RequestsCookieJar: ...


@overload
def cookiejar_from_dict(
    cookie_dict: dict[str, str] | None,
    cookiejar: _CookieJarT,
    overwrite: bool = True,
) -> _CookieJarT: ...


def cookiejar_from_dict(
    cookie_dict: dict[str, str] | None,
    cookiejar: CookieJar | None = None,
    overwrite: bool = True,
) -> CookieJar:
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    :rtype: CookieJar
    """
    if cookiejar is None:
        cookiejar = RequestsCookieJar()

    if cookie_dict is not None:
        names_from_jar = [cookie.name for cookie in cookiejar]
        for name in cookie_dict:
            if overwrite or (name not in names_from_jar):
                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))

    return cookiejar


def merge_cookies(
    cookiejar: CookieJar, cookies: dict[str, str] | CookieJar | None
) -> CookieJar:
    """Add cookies to cookiejar and returns a merged CookieJar.

    :param cookiejar: CookieJar object to add the cookies to.
    :param cookies: Dictionary or CookieJar object to be added.
    :rtype: CookieJar
    """
    if not isinstance(cookiejar, cookielib.CookieJar):  # runtime guard
        raise ValueError("You can only merge into CookieJar")

    if isinstance(cookies, dict):
        cookiejar = cookiejar_from_dict(cookies, cookiejar=cookiejar, overwrite=False)
    elif isinstance(cookies, cookielib.CookieJar):
        if update_method := getattr(cookiejar, "update", None):
            update_method(cookies)
        else:
            for cookie_in_jar in cookies:
                cookiejar.set_cookie(cookie_in_jar)

    return cookiejar
