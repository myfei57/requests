"""Tests for the local PreparedRequest.fingerprint() interface."""

import hashlib
import io
import re
import threading
from urllib.parse import urlsplit, urlunsplit

import pytest

import requests
from tests.testserver.server import Server, consume_socket_content

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prepared(method="GET", url="https://example.com/path", **kwargs):
    return requests.Request(method, url, **kwargs).prepare()


def _frame(tag, data):
    """Independent re-implementation of the length-framed field encoding."""
    return tag + len(data).to_bytes(8, "big") + data


def expected_fingerprint(method, url, selected_headers, header_values, body):
    """Compute the expected digest without calling the code under test.

    ``header_values`` maps lowercase header names to their bytes/str value,
    or omits a name entirely to mark it as missing.
    """
    hasher = hashlib.sha256()
    hasher.update(b"requests.PreparedRequest.fingerprint\x00v1\n")

    if method is None:
        hasher.update(b"m")
    else:
        hasher.update(_frame(b"M", method.encode("utf-8")))

    parts = urlsplit(url)
    without_fragment = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, parts.query, "")
    )
    hasher.update(_frame(b"U", without_fragment.encode("utf-8")))

    names = sorted({name.lower() for name in selected_headers})
    hasher.update(b"H")
    hasher.update(len(names).to_bytes(8, "big"))
    for name in names:
        hasher.update(_frame(b"h", name.encode("utf-8")))
        if name in header_values:
            value = header_values[name]
            if isinstance(value, str):
                value = value.encode("utf-8")
            hasher.update(_frame(b"P", value))
        else:
            hasher.update(b"A")

    if body is None:
        hasher.update(b"n")
    else:
        hasher.update(b"b")
        hasher.update(body)

    return hasher.hexdigest()


class SeekableStream:
    """A seekable binary stream with instrumentation."""

    def __init__(self, data, chunk_size=None, fail_on_read=None):
        self._buf = io.BytesIO(data)
        self.chunk_size = chunk_size
        self.fail_on_read = fail_on_read
        self.reads = 0
        self.seek_calls = []

    def read(self, size=-1):
        self.reads += 1
        if self.fail_on_read is not None and self.reads >= self.fail_on_read:
            raise RuntimeError("boom")
        if self.chunk_size is not None:
            size = min(size, self.chunk_size)
        return self._buf.read(size)

    def seek(self, offset, whence=io.SEEK_SET):
        self.seek_calls.append((offset, whence))
        return self._buf.seek(offset, whence)

    def tell(self):
        return self._buf.tell()

    def seekable(self):
        return True


class NonSeekableStream:
    """A binary stream that reports itself as non-seekable."""

    def __init__(self):
        self.read_called = False

    def read(self, size=-1):
        self.read_called = True
        return b""

    def seekable(self):
        return False

    def seek(self, offset, whence=io.SEEK_SET):
        raise OSError("not seekable")

    def tell(self):
        raise OSError("not seekable")


class NoPositionStream:
    """A readable stream without seek()/tell() support."""

    def __init__(self):
        self.read_called = False

    def read(self, size=-1):
        self.read_called = True
        return b""


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def test_fingerprint_returns_sha256_hex_digest():
    preq = make_prepared()
    digest = preq.fingerprint()

    assert isinstance(digest, str)
    assert DIGEST_RE.fullmatch(digest)
    assert digest == expected_fingerprint(
        "GET", "https://example.com/path", [], {}, None
    )


def test_fingerprint_is_deterministic_and_equals_for_equal_requests():
    preq1 = make_prepared(
        "POST",
        "https://example.com/api",
        headers={"Content-Type": "application/json"},
        data=b'{"a": 1}',
    )
    preq2 = make_prepared(
        "POST",
        "https://example.com/api",
        headers={"Content-Type": "application/json"},
        data=b'{"a": 1}',
    )

    assert preq1.fingerprint(headers=["Content-Type"]) == preq2.fingerprint(
        headers=["Content-Type"]
    )
    assert preq1.fingerprint(headers=["Content-Type"]) == preq1.fingerprint(
        headers=["Content-Type"]
    )


def test_fingerprint_matches_independently_built_digest():
    preq = make_prepared(
        "POST",
        "https://example.com/path?a=1&a=2#section",
        headers={"X-Empty": "", "Content-Type": "application/json"},
        data=b'{"x": 1}',
    )
    selected = ["Content-Type", "X-Empty", "X-Missing"]

    assert preq.fingerprint(headers=selected) == expected_fingerprint(
        "POST",
        "https://example.com/path?a=1&a=2#section",
        selected,
        {"content-type": "application/json", "x-empty": ""},
        b'{"x": 1}',
    )


def test_fingerprint_format_has_a_pinned_digest():
    # Locks the exact versioned framing.  Any change to the encoding must be
    # a deliberate, versioned change.
    preq = make_prepared(
        "POST",
        "https://example.com/path?a=1&a=2#section",
        headers={"X-Empty": "", "Content-Type": "application/json"},
        data=b'{"x": 1}',
    )
    assert (
        preq.fingerprint(headers=["Content-Type", "X-Empty", "X-Missing"])
        == "2285dd2eb4b262c1e12d4ba96ac20195c7c028eb9f562214f1c17f59ebf0a756"
    )


def test_different_methods_differ():
    get = make_prepared("GET", "https://example.com/")
    post = make_prepared("POST", "https://example.com/")

    assert get.fingerprint() != post.fingerprint()


def test_field_boundaries_are_unambiguous():
    # With naive concatenation the header name+value pairs would both be
    # "abc"; length framing keeps them distinct.
    first = make_prepared("GET", "https://example.com/", headers={"a": "bc"})
    second = make_prepared("GET", "https://example.com/", headers={"ab": "c"})

    assert first.fingerprint(headers=["a"]) != second.fingerprint(headers=["ab"])


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


def test_fragment_is_removed():
    with_fragment = make_prepared("GET", "https://example.com/path#frag")
    without_fragment = make_prepared("GET", "https://example.com/path")

    assert with_fragment.fingerprint() == without_fragment.fingerprint()


def test_empty_fragment_marker_is_removed():
    assert (
        make_prepared("GET", "https://example.com/path#").fingerprint()
        == make_prepared("GET", "https://example.com/path").fingerprint()
    )


def test_encoded_hash_in_query_is_preserved():
    # "%23" is data inside the query string, not a fragment separator.
    preq = make_prepared("GET", "https://example.com/path?q=%23x")
    assert preq.fingerprint() == expected_fingerprint(
        "GET", "https://example.com/path?q=%23x", [], {}, None
    )


def test_duplicate_query_parameters_are_preserved_in_order():
    preq = make_prepared("GET", "https://example.com/?a=1&a=2")
    assert preq.fingerprint() == expected_fingerprint(
        "GET", "https://example.com/?a=1&a=2", [], {}, None
    )

    # Same values built through params keep their order and duplicates.
    preq_params = make_prepared(
        "GET", "https://example.com/", params=[("a", 1), ("a", 2)]
    )
    assert preq_params.fingerprint() == preq.fingerprint()


def test_query_parameter_order_and_duplicates_matter():
    one_two = make_prepared("GET", "https://example.com/?a=1&a=2")
    two_one = make_prepared("GET", "https://example.com/?a=2&a=1")
    single = make_prepared("GET", "https://example.com/?a=1")

    assert one_two.fingerprint() != two_one.fingerprint()
    assert one_two.fingerprint() != single.fingerprint()


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------


def test_selected_header_names_are_case_insensitive():
    preq = make_prepared(
        "GET", "https://example.com/", headers={"Content-Type": "text/plain"}
    )

    assert preq.fingerprint(headers=["content-type"]) == preq.fingerprint(
        headers=["CONTENT-TYPE"]
    )
    # Duplicate names differing only in case are de-duplicated.
    assert preq.fingerprint(
        headers=["content-type", "Content-Type", "CONTENT-TYPE"]
    ) == preq.fingerprint(headers=["content-type"])


def test_prepared_header_casing_does_not_matter():
    lower = make_prepared(
        "GET", "https://example.com/", headers={"content-type": "text/plain"}
    )
    upper = make_prepared(
        "GET", "https://example.com/", headers={"CONTENT-TYPE": "text/plain"}
    )

    assert lower.fingerprint(headers=["Content-Type"]) == upper.fingerprint(
        headers=["content-type"]
    )


def test_header_selection_order_is_normalized():
    preq = make_prepared(
        "GET",
        "https://example.com/",
        headers={"A": "1", "B": "2", "C": "3"},
    )

    assert preq.fingerprint(headers=["c", "a", "b"]) == preq.fingerprint(
        headers=["A", "B", "C"]
    )


def test_missing_header_differs_from_empty_value_header():
    empty_value = make_prepared("GET", "https://example.com/", headers={"X-Token": ""})
    missing = make_prepared("GET", "https://example.com/")

    assert empty_value.fingerprint(headers=["X-Token"]) != missing.fingerprint(
        headers=["X-Token"]
    )


def test_empty_value_is_present_not_missing():
    preq = make_prepared("GET", "https://example.com/", headers={"X-Token": ""})
    assert preq.fingerprint(headers=["X-Token"]) == expected_fingerprint(
        "GET",
        "https://example.com/",
        ["X-Token"],
        {"x-token": b""},
        None,
    )


def test_missing_selected_header_is_recorded_as_absent():
    preq = make_prepared("GET", "https://example.com/", headers={"Other": "x"})
    assert preq.fingerprint(headers=["X-Token"]) == expected_fingerprint(
        "GET", "https://example.com/", ["X-Token"], {}, None
    )


def test_unselected_headers_do_not_affect_the_fingerprint():
    base = make_prepared("GET", "https://example.com/", headers={"X-Keep": "same"})
    with_extra = make_prepared(
        "GET",
        "https://example.com/",
        headers={"X-Keep": "same", "X-Ignored": "whatever"},
    )

    assert base.fingerprint(headers=["X-Keep"]) == with_extra.fingerprint(
        headers=["X-Keep"]
    )
    # The same is true with no headers selected at all.
    assert base.fingerprint() == with_extra.fingerprint()
    # But selecting the differing header breaks the tie.
    assert base.fingerprint(headers=["X-Keep", "X-Ignored"]) != with_extra.fingerprint(
        headers=["X-Keep", "X-Ignored"]
    )


def test_no_headers_selected_by_default():
    preq = make_prepared(
        "GET", "https://example.com/", headers={"X-A": "1", "X-B": "2"}
    )
    assert preq.fingerprint() == preq.fingerprint(headers=[])
    assert preq.fingerprint() == expected_fingerprint(
        "GET", "https://example.com/", [], {}, None
    )


def test_bytes_header_values_match_str_values():
    as_str = make_prepared("GET", "https://example.com/", headers={"X-Token": "abc"})
    as_bytes = make_prepared("GET", "https://example.com/", headers={"X-Token": b"abc"})
    assert as_str.fingerprint(headers=["X-Token"]) == as_bytes.fingerprint(
        headers=["X-Token"]
    )


def test_non_string_header_name_is_rejected():
    preq = make_prepared()
    with pytest.raises(TypeError):
        preq.fingerprint(headers=[b"content-type"])
    with pytest.raises(TypeError):
        preq.fingerprint(headers=[42])


# ---------------------------------------------------------------------------
# Body: bytes / None
# ---------------------------------------------------------------------------


def test_none_body_differs_from_empty_bytes_body():
    none_body = make_prepared("POST", "https://example.com/")
    empty_body = make_prepared("POST", "https://example.com/")
    empty_body.body = b""

    assert none_body.fingerprint() != empty_body.fingerprint()
    assert empty_body.fingerprint() == expected_fingerprint(
        "POST", "https://example.com/", [], {}, b""
    )


def test_bytes_body_is_hashed():
    preq = make_prepared("POST", "https://example.com/", data=b"abc")
    assert preq.fingerprint() == expected_fingerprint(
        "POST", "https://example.com/", [], {}, b"abc"
    )


def test_str_body_is_rejected_unchanged():
    # A prepared form-encoded request leaves a str body; callers must encode.
    preq = make_prepared("POST", "https://example.com/", data={"field": "value"})
    assert isinstance(preq.body, str)
    with pytest.raises(TypeError):
        preq.fingerprint()


# ---------------------------------------------------------------------------
# Body: seekable binary streams
# ---------------------------------------------------------------------------


def test_seekable_stream_is_read_from_current_position_and_restored():
    data = b"hello world"
    stream = SeekableStream(data)
    stream.seek(6)

    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    # Only bytes from the current position are hashed, matching an equivalent
    # bytes body (the bytes that would actually be sent).
    assert preq.fingerprint() == expected_fingerprint(
        "POST", "https://example.com/", [], {}, b"world"
    )
    # The original position is restored...
    assert stream.tell() == 6
    # ...and the remaining body is still available for sending.
    assert stream.read() == b"world"


def test_seekable_stream_fingerprint_is_independent_of_chunk_boundaries():
    whole = SeekableStream(b"abcdefghij")
    chunked = SeekableStream(b"abcdefghij", chunk_size=3)

    preq_whole = make_prepared("PUT", "https://example.com/upload")
    preq_whole.body = whole
    preq_chunked = make_prepared("PUT", "https://example.com/upload")
    preq_chunked.body = chunked

    assert preq_whole.fingerprint() == preq_chunked.fingerprint()
    assert preq_whole.fingerprint() == expected_fingerprint(
        "PUT", "https://example.com/upload", [], {}, b"abcdefghij"
    )
    assert whole.tell() == 0
    assert chunked.tell() == 0


def test_fingerprinting_stream_twice_is_stable_and_keeps_position():
    stream = SeekableStream(b"x" * 30000, chunk_size=4096)
    stream.seek(123)

    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    first = preq.fingerprint()
    assert stream.tell() == 123
    second = preq.fingerprint()
    assert stream.tell() == 123
    assert first == second
    assert first == expected_fingerprint(
        "POST", "https://example.com/", [], {}, b"x" * (30000 - 123)
    )


def test_real_binary_file_nonzero_position(tmp_path):
    payload = b"0123456789" * 5000
    path = tmp_path / "body.bin"
    path.write_bytes(payload)

    with path.open("rb") as fh:
        fh.seek(100)
        preq = make_prepared("POST", "https://example.com/upload")
        preq.body = fh

        assert preq.fingerprint() == expected_fingerprint(
            "POST",
            "https://example.com/upload",
            [],
            {},
            payload[100:],
        )
        assert fh.tell() == 100
        assert fh.read(3) == b"012"


def test_read_failure_propagates_and_restores_position():
    stream = SeekableStream(b"abcdefghij", chunk_size=3, fail_on_read=2)
    stream.seek(4)

    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    with pytest.raises(RuntimeError, match="boom"):
        preq.fingerprint()

    # Position was restored even though reading failed.
    assert stream.seek_calls[-1] == (4, io.SEEK_SET)
    assert stream.tell() == 4


def test_text_stream_is_rejected_and_restored():
    stream = io.StringIO("text body")
    stream.seek(2)

    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    with pytest.raises(TypeError):
        preq.fingerprint()
    assert stream.tell() == 2


# ---------------------------------------------------------------------------
# Body: unsupported types are rejected before consumption
# ---------------------------------------------------------------------------


def test_non_seekable_stream_is_rejected_before_reading():
    stream = NonSeekableStream()
    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    with pytest.raises(TypeError):
        preq.fingerprint()
    assert stream.read_called is False


def test_stream_without_seek_and_tell_is_rejected_before_reading():
    stream = NoPositionStream()
    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    with pytest.raises(TypeError):
        preq.fingerprint()
    assert stream.read_called is False


def test_iterator_body_is_rejected_without_being_consumed():
    consumed = []

    def gen():
        for chunk in (b"a", b"b", b"c"):
            consumed.append(chunk)
            yield chunk

    preq = make_prepared("POST", "https://example.com/")
    preq.body = gen()

    with pytest.raises(TypeError):
        preq.fingerprint()
    assert consumed == []


def test_plain_iterable_body_is_rejected():
    preq = make_prepared("POST", "https://example.com/")
    preq.body = iter([b"a", b"b"])

    with pytest.raises(TypeError):
        preq.fingerprint()


def test_unsupported_body_type_is_rejected():
    preq = make_prepared("POST", "https://example.com/")
    preq.body = 12345

    with pytest.raises(TypeError):
        preq.fingerprint()


def test_stream_whose_tell_fails_is_rejected_before_reading():
    class BadTell(SeekableStream):
        def tell(self):
            raise OSError("no tell")

    stream = BadTell(b"abc")
    preq = make_prepared("POST", "https://example.com/")
    preq.body = stream

    with pytest.raises(TypeError):
        preq.fingerprint()
    assert stream.reads == 0


def test_stream_whose_seekable_check_fails_is_rejected():
    class BadSeekable:
        def read(self, size=-1):
            raise AssertionError("must not be read")

        def seekable(self):
            raise OSError("cannot tell")

        def seek(self, offset, whence=io.SEEK_SET):
            pass

        def tell(self):
            return 0

    preq = make_prepared("POST", "https://example.com/")
    preq.body = BadSeekable()

    with pytest.raises(TypeError):
        preq.fingerprint()


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------


def test_fingerprint_does_not_modify_prepared_request():
    preq = make_prepared(
        "POST",
        "https://example.com/?z=9",
        headers={"X-A": "1", "X-B": "2"},
        data=b"payload",
    )
    before = (
        preq.method,
        preq.url,
        list(preq.headers.items()),
        preq.body,
        preq._body_position,
    )

    preq.fingerprint(headers=["X-A", "X-Missing"])

    after = (
        preq.method,
        preq.url,
        list(preq.headers.items()),
        preq.body,
        preq._body_position,
    )
    assert after == before


def test_fingerprint_does_not_invoke_auth_or_hooks():
    class CountingAuth(requests.auth.AuthBase):
        def __init__(self):
            self.calls = 0

        def __call__(self, r):
            self.calls += 1
            r.headers["X-Auth"] = "token"
            return r

    fired_hooks = []
    auth = CountingAuth()
    preq = requests.Request(
        "GET",
        "https://example.com/",
        auth=auth,
        hooks={"response": [lambda r, *a, **k: fired_hooks.append(r)]},
    ).prepare()

    assert auth.calls == 1  # applied once, during prepare
    preq.fingerprint(headers=["X-Auth"])
    assert auth.calls == 1  # fingerprinting does not re-run authentication
    assert fired_hooks == []


def test_fingerprint_does_not_send_request(monkeypatch):
    def fail_send(*args, **kwargs):
        raise AssertionError("fingerprint() must not send the request")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fail_send)

    preq = make_prepared("GET", "https://example.com/")
    preq.fingerprint(headers=[])  # must return without any adapter activity


def test_unprepared_request_without_url_is_rejected():
    preq = requests.PreparedRequest()
    preq.prepare_method("GET")

    with pytest.raises(ValueError):
        preq.fingerprint()


# ---------------------------------------------------------------------------
# Integration with the send path (local socket server, no network)
# ---------------------------------------------------------------------------


def echo_handler(sock):
    request_content = consume_socket_content(sock, timeout=0.5)
    sock.send(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: %d\r\n\r\n"
        b"%s" % (len(request_content), request_content)
    )


def test_stream_body_can_still_be_sent_after_fingerprinting():
    close_server = threading.Event()
    server = Server(echo_handler, wait_to_close_event=close_server)

    # Seek past the first bytes before preparing: only the suffix would be
    # sent, and fingerprinting must not disturb that.
    stream = SeekableStream(b"hello world", chunk_size=3)
    stream.seek(6)

    with server as (host, port):
        url = f"http://{host}:{port}/"
        session = requests.Session()
        preq = session.prepare_request(requests.Request("POST", url, data=stream))

        # Equivalent to an explicit bytes body containing only the suffix.
        bytes_preq = session.prepare_request(
            requests.Request("POST", url, data=b"world")
        )
        assert preq.fingerprint() == bytes_preq.fingerprint()
        assert stream.tell() == 6

        response = session.send(preq)
        assert response.status_code == 200
        assert response.content.endswith(b"world")
        assert stream.tell() == 11

        close_server.set()
