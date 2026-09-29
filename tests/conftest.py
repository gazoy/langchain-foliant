"""A metered x402 API served in-process, and a funded Foliant agent, for every test."""
import sys
import warnings

import pytest
from fastapi.testclient import TestClient
from foliant import Agent, KeyPair, Ledger, Policy, ServiceOffer

warnings.filterwarnings("ignore", message=".*httpx2.*")

ASSET = "USDC"


def build_world(price: int = 3, cap: int = 200):
    """Ledger, provider API (in-process), and a funded agent with a policy of `cap` per hour."""
    from foliant.x402 import PaymentGate, install
    from fastapi import Depends, FastAPI, Request, Response

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

    http = TestClient(app, base_url="http://api")
    agent = Agent(L, KeyPair.from_seed(b"owner"), KeyPair.from_seed(b"signer"),
                  Policy(per_tx_max=500, per_window_max=cap, window_secs=3600))
    L.mint(agent.account.address, ASSET, 10_000)
    return L, http, agent, gate


@pytest.fixture
def world():
    return build_world()
