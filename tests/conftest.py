"""A metered x402 API served over loopback, and a funded Foliant agent, for every test."""
import socket
import threading
import time
import warnings

import httpx
import pytest
import uvicorn
from foliant import Agent, KeyPair, Ledger, Policy, ServiceOffer

warnings.filterwarnings("ignore", message=".*httpx2.*")

ASSET = "USDC"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serve(app) -> httpx.URL:
    """Run `app` on a loopback port in a daemon thread and return its base URL.

    `FoliantPaymentTool.http` is an `httpx.Client`, and starlette's `TestClient` stopped being one
    (it no longer subclasses it), so pydantic rejects it. Driving an ASGI app through a *sync*
    `httpx.Client` is not possible either, because `httpx.ASGITransport` is async-only. So the
    suite talks to a real server over loopback, which is also how the tool is used in anger. The
    thread is a daemon: the server lives as long as the test process and needs no teardown.
    """
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        time.sleep(0.05)
        if server.started:
            return httpx.URL(f"http://127.0.0.1:{port}")
    raise RuntimeError(f"the metered API did not start on 127.0.0.1:{port} within 10s")


class _ToLoopback(httpx.BaseTransport):
    """Sends every request to the test server, whatever host the caller named.

    The suite's URLs read `http://api/infer`, which `TestClient` used to intercept. A real client
    would try to resolve the host `api`, so this rewrites scheme, host and port to the loopback
    server and leaves the path, headers and body alone. Keeping the URLs as they were means the
    tests still read as the agent being handed an ordinary URL.
    """

    def __init__(self, base: httpx.URL) -> None:
        self._base = base
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.url = request.url.copy_with(
            scheme=self._base.scheme, host=self._base.host, port=self._base.port
        )
        request.headers["host"] = self._base.netloc.decode()
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def build_world(price: int = 3, cap: int = 200):
    """Ledger, provider API (over loopback), and a funded agent with a policy of `cap` per hour."""
    from fastapi import Depends, FastAPI, Request, Response
    from foliant.x402 import PaymentGate, install

    L = Ledger()
    provider = KeyPair.from_seed(b"provider")
    L.mint(provider.address, ASSET, 1_000)
    pool = L.create_pool(provider.address, ASSET, timeout_secs=60, bond=1_000)
    offer = ServiceOffer(provider=provider.address, asset=ASSET, price_per_unit=price, unit="call",
                         descriptor={"model": "echo-1"}, pool_id=pool.id)
    L.publish_offer(offer)
    gate = PaymentGate(L, offer, provider)
    app = FastAPI()
    install(app)

    @app.post("/infer")
    async def infer(request: Request, payment=Depends(gate.dependency())):
        body = await request.body()
        out = Response(content=b'{"echo":"' + body + b'"}', media_type="application/json")
        out.headers["X-PAYMENT-RESPONSE"] = gate.receipt(payment, body, out.body)
        return out

    http = httpx.Client(transport=_ToLoopback(_serve(app)), base_url="http://api", timeout=20)
    agent = Agent(L, KeyPair.from_seed(b"owner"), KeyPair.from_seed(b"signer"),
                  Policy(per_tx_max=500, per_window_max=cap, window_secs=3600))
    L.mint(agent.account.address, ASSET, 10_000)
    return L, http, agent, gate


@pytest.fixture
def world():
    return build_world()
