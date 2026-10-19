from __future__ import annotations

import asyncio

import httpx


def _request(app, method: str, path: str) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path)

    return asyncio.run(run())


class _FakeOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        calls.append("init")

    def initialize(self, reporter=None) -> None:
        calls.append("initialize")

    def set_realtime_event_sink(self, sink) -> None:
        pass

    def set_runtime_snapshot_provider(self, provider) -> None:
        pass

    def configure_mcp_control(self, url: str) -> None:
        pass

    def recover_interrupted_execution(self) -> None:
        calls.append("recover")

    def get_state(self) -> dict:
        return {"ok": True}

    def close(self) -> None:
        pass


calls: list[str] = []


def test_experiments_bootstrap_does_not_construct_the_legacy_harness(tmp_path, monkeypatch) -> None:
    import harnessing_ts.server as server

    calls.clear()
    monkeypatch.setattr(server, "HarnessOrchestrator", _FakeOrchestrator)
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("TS_HARNESS_LEGACY", raising=False)

    app = server.create_app()
    assert calls == []

    response = _request(app, "GET", "/api/bootstrap")

    assert response.status_code == 200
    assert response.json()["variant"] is None
    assert calls == []


def test_a_legacy_route_initializes_the_harness_on_first_use(tmp_path, monkeypatch) -> None:
    import harnessing_ts.server as server

    calls.clear()
    monkeypatch.setattr(server, "HarnessOrchestrator", _FakeOrchestrator)
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("TS_HARNESS_LEGACY", raising=False)

    response = _request(server.create_app(), "GET", "/api/state")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls[:3] == ["init", "initialize", "recover"]


def test_legacy_mode_still_initializes_at_startup(tmp_path, monkeypatch) -> None:
    import harnessing_ts.server as server

    calls.clear()
    monkeypatch.setattr(server, "HarnessOrchestrator", _FakeOrchestrator)
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TS_HARNESS_LEGACY", "true")

    server.create_app()

    assert calls[:3] == ["init", "initialize", "recover"]
