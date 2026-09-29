# langchain-foliant

[Foliant](https://foliant.network) for LangChain: a payment tool for agents that pay x402 endpoints, and a middleware that enforces the agent's spending budget on every paid tool call. Built for the supervisor / subagents pattern: the supervisor holds the crew's budget, each subagent gets a delegated cap, and every spend is checked against every level.

```bash
pip install langchain-foliant   # pulls in foliant-protocol, the reference implementation
```

## What it does

- **`FoliantPaymentTool`** — calls an HTTP endpoint that charges per request over x402. On a 402 it opens a payment channel to, or joins the pool of, the provider, then signs one off-chain update per call. Thousands of calls settle in one on-chain transaction. If the agent's policy does not allow the payment the tool returns `BUDGET REFUSED: …` to the model rather than raising.
- **`FoliantBudgetMiddleware`** — wraps tool calls; before a paid call it checks that the agent can still pay (policy headroom plus unspent balance in open channels or pools) and returns a clear tool message, or stops the run, if not.
- **`FoliantCrew`** — maps a supervisor and its subagents onto a Foliant budget tree. `worker(name, policy, fund=…)` delegates a child account with a policy no wider than the supervisor's and returns the tool and middleware for that subagent; `revoke(name)` expires the worker and recalls its balance with the supervisor's key alone.

The wire format is unchanged x402 (`X-PAYMENT`, `X-PAYMENT-RESPONSE`, the 402 `accepts` body), so this sits beside `langchain-x402` and per-call wallets rather than replacing them.

## Example

```python
import httpx
from foliant import Agent, KeyPair, Ledger, Policy
from langchain.agents import create_agent
from langchain_foliant import FoliantCrew

ledger = Ledger()                       # the reference in-memory ledger; contracts on Base/Avalanche are the next phase
http = httpx.Client()

supervisor = Agent(ledger, KeyPair(), KeyPair(),
                   Policy(per_tx_max=500, per_window_max=150, window_secs=3600))
crew = FoliantCrew(supervisor, http)

worker_policy = Policy(per_tx_max=100, per_window_max=100, window_secs=3600)
research = crew.worker("research", worker_policy, fund=1_000)
writer = crew.worker("writer", worker_policy, fund=1_000)

research_agent = create_agent("openai:gpt-5.5", tools=[research.tool], middleware=[research.middleware])
writer_agent = create_agent("openai:gpt-5.5", tools=[writer.tool], middleware=[writer.middleware])

# Each worker may commit up to 100 in the hour; the crew as a whole stops at 150,
# so the third worker's deposit is refused even though its own cap has room.
# crew.committed() -> {"research": 100, "writer": 50, "crew": 150}
# crew.revoke("writer") -> expires the writer's policy and recalls its unspent balance.
```

## Refund and settlement semantics

Deposits into a channel or pool are the committed value the policy sees; per-call updates are bounded by the deposit. The provider settles at its own cadence with one transaction per channel or per pool. A payer recovers its unspent deposit by closing a channel or exiting a pool unilaterally after the timeout; a provider that goes silent cannot hold a payer's funds. Receipts are provider-signed, one per call, and are available on the tool as `tool.receipts`.

## Status

The tool and middleware run against the Foliant Python reference implementation (an in-process ledger) today, which is what the tests exercise. Solidity contracts and a registered x402 scheme on Base and Avalanche are the next phase; the tool's interface does not change when they land.

## Tests

```bash
pip install -e ".[test]"
pytest
```

Includes LangChain's standard tool tests (`langchain-tests`) and behaviour tests for payment, refusal, middleware inside `create_agent`, and the crew.

## Licence

Apache-2.0. Copyright 2026 Machine Quotient Ltd.
