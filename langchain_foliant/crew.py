"""FoliantCrew: map LangChain's supervisor / subagents pattern onto a Foliant budget tree."""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from foliant import Agent, KeyPair, Policy

from .middleware import FoliantBudgetMiddleware
from .tools import FoliantPaymentTool


@dataclass
class Worker:
    name: str
    agent: Agent
    tool: FoliantPaymentTool
    middleware: FoliantBudgetMiddleware


@dataclass
class FoliantCrew:
    """A supervisor's Foliant account plus one delegated account per subagent.

    The supervisor holds the crew's budget. `worker(name, policy)` delegates a
    child account with a policy no wider than the supervisor's, funds it, and
    returns the tool and middleware to give that subagent. Every worker's spend
    is checked against its own policy and the supervisor's, so no worker can
    exceed its cap and the crew cannot exceed the supervisor's. `revoke(name)`
    expires the worker's policy and recalls its balance, with the supervisor's
    key alone.
    """

    supervisor: Agent
    http: httpx.Client
    asset: str = "USDC"
    default_deposit: int = 100
    prefer_pool: bool = True
    workers: dict[str, Worker] = field(default_factory=dict)

    def worker(self, name: str, policy: Policy, *, fund: int, signer: KeyPair | None = None) -> Worker:
        if name in self.workers:
            raise ValueError(f"worker {name!r} exists")
        kp = signer or KeyPair.from_seed(f"{self.supervisor.account.id}:{name}".encode())
        agent = self.supervisor.delegate(kp, policy, fund=fund, asset=self.asset, salt=len(self.workers))
        tool = FoliantPaymentTool(agent=agent, http=self.http, default_deposit=self.default_deposit,
                                  prefer_pool=self.prefer_pool)
        mw = FoliantBudgetMiddleware(agent)
        w = Worker(name, agent, tool, mw)
        self.workers[name] = w
        return w

    def revoke(self, name: str) -> int:
        """Expire the worker's policy and recall its unspent balance. Returns the amount recalled."""
        w = self.workers[name]
        pol = w.agent.signer.policy
        dead = Policy(per_tx_max=0, per_window_max=0, window_secs=pol.window_secs,
                      allow_list=pol.allow_list, deny_list=pol.deny_list, expiry=self.supervisor.ledger.now)
        self.supervisor.set_child_policy(w.agent, dead)
        return self.supervisor.recall(w.agent, self.asset)["recalled"]

    def committed(self) -> dict[str, int]:
        """Committed value in the current window, per worker and for the crew."""
        L = self.supervisor.ledger
        out = {n: w.agent.account.window.spent(L.now, w.agent.account.policy.window_secs) for n, w in self.workers.items()}
        out["crew"] = self.supervisor.account.window.spent(L.now, self.supervisor.account.policy.window_secs)
        return out
