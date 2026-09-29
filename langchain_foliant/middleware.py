"""FoliantBudgetMiddleware: enforce a Foliant spending policy around an agent's tool calls."""
from __future__ import annotations

from typing import Callable, Literal

from foliant import Agent
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class BudgetExhaustedError(RuntimeError):
    pass


class FoliantBudgetMiddleware(AgentMiddleware):
    """Refuse paid tool calls once the agent can no longer pay.

    The signer and the ledger already refuse over-budget payments inside the
    tool; this middleware makes the refusal visible before the call, so the
    model gets a clear tool message (or the run stops) instead of a failed
    request. "Can pay" means: the agent's own policy allows a new deposit in the
    current window, or an open channel or pool claim still has unspent balance.
    The ledger also checks every ancestor's policy when a deposit is made, so a
    crew-level refusal still surfaces from the tool as "BUDGET REFUSED".

    Args:
        agent: the Foliant agent whose budget applies.
        tool_name: only tool calls with this name are checked (default: the
            payment tool, "foliant_pay"). Pass None to check every tool call.
        min_headroom: the smallest amount the agent must still be able to pay
            for a call to proceed; set it to the endpoint's price to refuse
            before an underfunded request rather than after.
        exit_behavior: "continue" returns a ToolMessage and lets the model carry
            on; "error" raises BudgetExhaustedError.
    """

    def __init__(self, agent: Agent, *, tool_name: str | None = "foliant_pay", min_headroom: int = 1,
                 exit_behavior: Literal["continue", "error"] = "continue") -> None:
        super().__init__()
        if exit_behavior not in ("continue", "error"):
            raise ValueError(f"Invalid exit_behavior: {exit_behavior!r}")
        self.agent = agent
        self.tool_name = tool_name
        self.min_headroom = min_headroom
        self.exit_behavior = exit_behavior

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}[{self.tool_name or '*'}]"

    def headroom(self) -> int:
        """Value the agent can still pay: what its policy allows as a new deposit in the
        current window, plus what is left unspent in its open channels and pool claims.
        A refusal from an ancestor's policy is not visible here; the tool reports it."""
        agent, L = self.agent, self.agent.ledger
        pol, now = agent.signer.policy, L.now
        if pol.expiry is not None and now >= pol.expiry:
            return 0
        room = max(0, pol.per_window_max - agent.signer.window.spent(now, pol.window_secs))
        for ch in L.channels.values():
            if ch.payer_account == agent.account.id and not ch.closed and ch.closing_at is None:
                owed = agent.latest[ch.id].body["balance"] if ch.id in agent.latest else ch.balance_to_payee
                room += ch.deposit - owed
        for pool in L.pools.values():
            claim = pool.members.get(agent.account.id)
            if claim is not None and not claim.exited and claim.exit_at is None:
                owed = agent.latest[pool.id].body["balance"] if pool.id in agent.latest else claim.paid
                room += claim.deposit - owed
        return room

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        call = request.tool_call
        if self.tool_name is not None and call["name"] != self.tool_name:
            return handler(request)
        room = self.headroom()
        if room >= self.min_headroom:
            return handler(request)
        msg = f"BUDGET REFUSED: {room} left in this window; policy allows {self.agent.signer.policy.per_window_max}"
        if self.exit_behavior == "error":
            raise BudgetExhaustedError(msg)
        return ToolMessage(content=msg, tool_call_id=call["id"], name=call["name"], status="error")

    async def awrap_tool_call(self, request: ToolCallRequest, handler):  # type: ignore[override]
        call = request.tool_call
        if self.tool_name is not None and call["name"] != self.tool_name:
            return await handler(request)
        room = self.headroom()
        if room >= self.min_headroom:
            return await handler(request)
        msg = f"BUDGET REFUSED: {room} left in this window; policy allows {self.agent.signer.policy.per_window_max}"
        if self.exit_behavior == "error":
            raise BudgetExhaustedError(msg)
        return ToolMessage(content=msg, tool_call_id=call["id"], name=call["name"], status="error")
