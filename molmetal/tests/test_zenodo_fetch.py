"""Tests for zenodo_fetch.py — DNS-resilient Zenodo file fetcher (TODO-33 P7).

Covers:
* DNS probe returns blocked status when ``0.0.0.0`` is returned.
* DNS probe returns ok status when a real address is returned.
* ``zenodo_fetch`` returns a structured ``ZenodoFetchResult``.
* All failure modes (DNS blocked + all IPs fail) end in
  ``ok=False, method='manual'`` with a manual-download message.
* Successful direct-DNS path is detected and ``method='direct-dns'``.
* Successful IP-pin path is detected and ``method='ip-pin'``.

Network-dependent tests are wrapped in the ``integration`` marker and
are skipped by default (``pytest -m 'not integration'``).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.zenodo_fetch import (
    ZenodoFetchResult,
    ZENODO_FALLBACK_IPS,
    _http_download_direct,
    _http_download_via_ip,
    _resolve_zenodo_dns,
    zenodo_fetch,
)


# ---------------------------------------------------------------------------
# DNS-probe tests (pure-python, no network)
# ---------------------------------------------------------------------------
class TestDnsProbe:
    def test_blocked_when_all_addrs_are_zero(self, monkeypatch):
        """If every resolved address is ``0.0.0.0``, return blocked."""

        def fake_getaddrinfo(*_args, **_kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", 443)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        ok, info = _resolve_zenodo_dns()
        assert ok is False
        assert "blocked" in info.lower() or "0.0.0.0" in info

    def test_ok_when_real_addr_returned(self, monkeypatch):
        """If the resolver returns a routable address, return ok=True."""

        def fake_getaddrinfo(*_args, **_kwargs):
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("159.100.250.205", 443),
                ),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        ok, info = _resolve_zenodo_dns()
        assert ok is True
        assert "159.100.250.205" in info

    def test_gaierror_returns_blocked(self, monkeypatch):
        """DNS failure surfaces as blocked, not exception."""

        def fake_getaddrinfo(*_args, **_kwargs):
            raise socket.gaierror("Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        ok, info = _resolve_zenodo_dns()
        assert ok is False
        assert "DNS resolution failed" in info


# ---------------------------------------------------------------------------
# HTTP-helper tests (pure-python, no network)
# ---------------------------------------------------------------------------
class TestHttpHelpers:
    def test_http_download_direct_returns_bytes(self, tmp_path, monkeypatch):
        """Direct DNS path should write the bytes and return ok=True."""
        fake_content = b"hello-zenodo-" * 1024
        captured = {}

        class _FakeResp:
            def __init__(self, body):
                self._body = body

            def read(self, n):
                if not self._body:
                    return b""
                chunk, self._body = self._body[:n], self._body[n:]
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_urlopen(url, **kwargs):
            captured["url"] = url
            return _FakeResp(fake_content)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        dst = tmp_path / "out.bin"
        ok, n, msg = _http_download_direct(
            "https://zenodo.org/records/123/files/foo.bin/content",
            dst,
            timeout=10,
        )
        assert ok is True
        assert n == len(fake_content)
        assert dst.exists()
        assert dst.read_bytes() == fake_content
        assert "zenodo.org" in captured["url"]

    def test_http_download_via_ip_sets_host_header(self, tmp_path, monkeypatch):
        """IP-pin path should pin the IP and set the Host header to zenodo.org."""
        captured = {}

        class _FakeResp:
            def __init__(self):
                pass

            def read(self, n):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_urlopen(req, **kwargs):
            captured["url"] = req.full_url
            captured["host_header"] = req.get_header("Host")
            return _FakeResp()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        dst = tmp_path / "out.bin"
        ok, n, msg = _http_download_via_ip(
            "zenodo.org",
            "/records/123/files/foo.bin/content",
            dst,
            pinned_ip="159.100.250.205",
            timeout=10,
        )
        assert ok is True
        assert captured["url"].startswith("https://159.100.250.205")
        assert captured["host_header"] == "zenodo.org"

    def test_http_download_via_ip_returns_zero_bytes_on_empty(
        self, tmp_path, monkeypatch
    ):
        """An empty body returns ok=True with 0 bytes written."""

        class _FakeResp:
            def read(self, n):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _FakeResp())
        dst = tmp_path / "out.bin"
        ok, n, msg = _http_download_via_ip(
            "zenodo.org",
            "/foo",
            dst,
            pinned_ip="159.100.250.205",
        )
        assert ok is True
        assert n == 0


# ---------------------------------------------------------------------------
# zenodo_fetch orchestration tests (no network — all helpers mocked)
# ---------------------------------------------------------------------------
class TestZenodoFetchOrchestration:
    def test_direct_dns_path_returns_ok(self, tmp_path, monkeypatch):
        """Happy path: DNS resolves → direct fetch → method=direct-dns."""
        # DNS probe says ok
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._resolve_zenodo_dns",
            lambda **_k: (True, "159.100.250.205"),
        )

        # Direct fetch returns ok with 13 bytes
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._http_download_direct",
            lambda url, dst, **_k: (True, 13, "13 bytes via direct DNS"),
        )

        # IP-pin and HEAD paths must NOT be called
        ip_pin_calls = []
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._http_download_via_ip",
            lambda *a, **k: ip_pin_calls.append(a)
            or (False, 0, "should-not-be-called"),
        )
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._head_via_ip",
            lambda *a, **k: (False, "should-not-be-called"),
        )

        result = zenodo_fetch("8183747", "foo.ckpt", tmp_path, verbose=False)
        assert result.ok is True
        assert result.method == "direct-dns"
        assert result.bytes_written == 13
        assert "direct DNS" in result.message
        assert ip_pin_calls == []
        assert (tmp_path / "foo.ckpt").exists() is False  # mocked; real fetch skipped

    def test_dns_blocked_then_ip_pin_succeeds(self, tmp_path, monkeypatch):
        """DNS blocked → first IP-pin succeeds → method=ip-pin."""
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._resolve_zenodo_dns",
            lambda **_k: (False, "DNS blocked — all addresses 0.0.0.0"),
        )
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._head_via_ip",
            lambda *a, **k: (True, "HTTP 200, content-length=1024"),
        )
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._http_download_via_ip",
            lambda *a, **k: (True, 1024, "1024 bytes via 159.100.250.205"),
        )
        # Suppress the doi.org probe — it's not relevant here
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch.urllib.request.urlopen",
            mock.Mock(side_effect=Exception("doi probe disabled in this test")),
        )

        result = zenodo_fetch(
            "8183747",
            "foo.ckpt",
            tmp_path,
            fallback_ips=("159.100.250.205", "159.100.250.206"),
        )
        assert result.ok is True
        assert result.method == "ip-pin"
        assert result.bytes_written == 1024
        assert "159.100.250.205" in result.message

    def test_all_paths_fail_returns_manual_method(self, tmp_path, monkeypatch):
        """DNS blocked + all IP-pins fail → method=manual with a URL hint."""
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._resolve_zenodo_dns",
            lambda **_k: (False, "DNS blocked"),
        )
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._head_via_ip",
            lambda *a, **k: (False, "URLError: connection refused"),
        )
        # IP-pin download would only be invoked if ip-head succeeded; mock
        # it anyway so the test catches any unexpected invocation.
        ip_get_called = []
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._http_download_via_ip",
            lambda *a, **k: ip_get_called.append(a)
            or (False, 0, "should-not-be-called"),
        )
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch.urllib.request.urlopen",
            mock.Mock(side_effect=Exception("doi blocked too")),
        )

        result = zenodo_fetch(
            "8183747",
            "foo.ckpt",
            tmp_path,
            fallback_ips=("159.100.250.205", "159.100.250.206"),
        )
        assert result.ok is False
        assert result.method == "manual"
        assert "Manual download" in result.message
        assert "8183747" in result.message
        # Attempt trace must mention dns-probe, ip-head-* (one per IP), and
        # doi-redirect. ip-get-* is only attempted if ip-head succeeded.
        steps = [a["step"] for a in result.attempts]
        assert "dns-probe" in steps
        assert sum(1 for s in steps if s.startswith("ip-head-")) == 2
        assert "doi-redirect" in steps
        # When ip-head fails for every IP, ip-get is never invoked
        assert ip_get_called == []

    def test_attempts_list_captures_all_steps(self, tmp_path, monkeypatch):
        """The attempts list is populated even on the happy path."""
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._resolve_zenodo_dns",
            lambda **_k: (True, "159.100.250.205"),
        )
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._http_download_direct",
            lambda url, dst, **_k: (True, 1, "1 bytes via direct DNS"),
        )
        result = zenodo_fetch("8183747", "foo.ckpt", tmp_path)
        assert result.ok is True
        step_names = [a["step"] for a in result.attempts]
        assert "dns-probe" in step_names
        assert "direct-dns" in step_names
        # No IP-pin steps on the happy path
        assert not any(s.startswith("ip-head-") for s in step_names)
        assert not any(s.startswith("ip-get-") for s in step_names)


# ---------------------------------------------------------------------------
# ZenodoFetchResult dataclass tests
# ---------------------------------------------------------------------------
class TestZenodoFetchResult:
    def test_to_dict_has_all_fields(self):
        r = ZenodoFetchResult(
            ok=True,
            record_id="1",
            filename="x",
            dst_path="/tmp/x",
            method="direct-dns",
            bytes_written=42,
            elapsed_s=0.5,
            message="ok",
        )
        d = r.to_dict()
        assert d["ok"] is True
        assert d["record_id"] == "1"
        assert d["method"] == "direct-dns"
        assert d["bytes_written"] == 42
        assert d["elapsed_s"] == 0.5
        assert "attempts" in d
        assert isinstance(d["attempts"], list)

    def test_json_round_trip(self):
        r = ZenodoFetchResult(
            ok=False,
            record_id="2",
            filename="y",
            dst_path=None,
            method="manual",
            bytes_written=0,
            elapsed_s=1.0,
            message="manual",
            attempts=[{"step": "dns-probe", "info": "blocked"}],
        )
        js = json.dumps(r.to_dict())
        loaded = json.loads(js)
        assert loaded["record_id"] == "2"
        assert loaded["method"] == "manual"
        assert loaded["attempts"][0]["step"] == "dns-probe"


# ---------------------------------------------------------------------------
# Module-level config tests
# ---------------------------------------------------------------------------
class TestConfig:
    def test_fallback_ips_is_nonempty_tuple(self):
        assert isinstance(ZENODO_FALLBACK_IPS, tuple)
        assert len(ZENODO_FALLBACK_IPS) >= 1
        for ip in ZENODO_FALLBACK_IPS:
            # Sanity: all IPs are valid IPv4 strings
            parts = ip.split(".")
            assert len(parts) == 4
            for p in parts:
                assert p.isdigit()
                assert 0 <= int(p) <= 255

    def test_main_help_exits_zero(self):
        """--help should exit 0 (not raise SystemExit)."""
        from molmetal.scripts.zenodo_fetch import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_main_probe_only_exits_zero(self, monkeypatch, capsys):
        """--probe-only should call the DNS probe and print status."""
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._resolve_zenodo_dns",
            lambda **_k: (True, "159.100.250.205"),
        )
        from molmetal.scripts.zenodo_fetch import main

        rc = main(["--probe-only"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "159.100.250.205" in out

    def test_main_probe_only_returns_one_when_blocked(self, monkeypatch):
        """--probe-only should exit 1 when DNS is blocked."""
        monkeypatch.setattr(
            "molmetal.scripts.zenodo_fetch._resolve_zenodo_dns",
            lambda **_k: (False, "DNS blocked"),
        )
        from molmetal.scripts.zenodo_fetch import main

        rc = main(["--probe-only"])
        assert rc == 1

    def test_main_requires_record_and_file(self, monkeypatch, capsys):
        """Without --record/--file and without --probe-only, exit 2."""
        from molmetal.scripts.zenodo_fetch import main

        rc = main([])
        assert rc == 2
        err = capsys.readouterr().err
        assert "--record" in err


# ---------------------------------------------------------------------------
# Optional network test (skipped unless MOLMETAL_RUN_NETWORK_TESTS=1)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MOLMETAL_RUN_NETWORK_TESTS"),
    reason="Network-dependent test — set MOLMETAL_RUN_NETWORK_TESTS=1 to enable",
)
def test_real_probe_to_zenodo():
    """Probe real zenodo.org DNS from this host.  Useful for diagnosis.

    Marked ``integration`` because it requires network egress to the
    Zenodo CDN.  Skipped unless ``MOLMETAL_RUN_NETWORK_TESTS=1`` is
    exported (same convention as ``test_round11_install_ladder.py``
    and other integration-marked tests).
    """
    ok, info = _resolve_zenodo_dns()
    # We don't assert ok=True because the host may legitimately have
    # zenodo blocked; just print the result.
    print(f"DNS probe: ok={ok} info={info}")
    # Either OK (we can fetch) or BLOCKED (we use IP-pin fallback).
    assert ok in (True, False)
