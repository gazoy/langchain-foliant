"""Behaviour: the tool pays, the budget refuses, the middleware and the crew do what they say."""
import pytest
from foliant import Policy
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_foliant import BudgetExhaustedError, FoliantBudgetMiddleware, FoliantCrew, FoliantPaymentTool

from tests.conftest import ASSET, build_world


def test_tool_pays_and_returns_receipt(world):
    L, http, agent, gate = world
    tool = FoliantPaymentTool(agent=agent, http=http)
    out = tool.invoke({"url": "http://api/infer", "method": "POST", "body": "hi"})
    assert out.startswith("200\n") and '"echo":"hi"' in out and "[receipt " in out
    assert len(tool.receipts) == 1
    for _ in range(9):
        tool.invoke({"url": "http://api/infer", "body": "x"})
    assert len(tool.receipts) == 10
    assert len([t for t in L.log]) <= 3  # one join (plus the pool bond); no per-call transactions
    assert gate.settle() == 30  # ten calls at 3, one settlement


def test_budget_refusal_is_returned_not_raised(world):
    L, http, agent, _ = world  # cap 200 per hour; deposits of 100
    tool = FoliantPaymentTool(agent=agent, http=http, default_deposit=100, prefer_pool=False)
    outs = [tool.invoke({"url": "http://api/infer", "body": "x"}) for _ in range(80)]
    refused = [o for o in outs if o.startswith("BUDGET REFUSED")]
    assert refused and "per_window_max 200" in refused[0]
    assert agent.account.window.spent(L.now, 3600) == 200 == agent.signer.window.spent(L.now, 3600)


class _ToolCallingFake(FakeMessagesListChatModel):
    """The stock fake returns scripted messages but cannot bind tools; accept and ignore them."""

    def bind_tools(self, tools, **kwargs):
        return self


def _model_that_calls(tool_name: str, n: int) -> FakeMessagesListChatModel:
    """A fake chat model that issues `n` tool calls, one per turn, then stops."""
    calls = [AIMessage(content="", tool_calls=[{"name": tool_name, "id": f"c{i}",
                                                 "args": {"url": "http://api/infer", "method": "POST", "body": f"q{i}"}}])
             for i in range(n)]
    return _ToolCallingFake(responses=calls + [AIMessage(content="done")])


def test_middleware_blocks_paid_calls_when_budget_is_gone(world):
    L, http, agent, _ = world
    agent.signer.policy = agent.account.policy = Policy(per_tx_max=500, per_window_max=100, window_secs=3600)
    tool = FoliantPaymentTool(agent=agent, http=http, default_deposit=100, prefer_pool=False)
    mw = FoliantBudgetMiddleware(agent)
    graph = create_agent(_model_that_calls(tool.name, 3), tools=[tool], middleware=[mw])
    result = graph.invoke({"messages": [HumanMessage("go")]})
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3
    assert all(t.content.startswith("200") for t in tool_msgs)  # the first call opens a 100-unit channel; the rest draw on it
    assert mw.headroom() == 100 - 9  # nothing left for a new deposit; 91 unspent in the channel


def test_middleware_refuses_before_the_tool_when_no_headroom(world):
    L, http, agent, _ = world
    agent.signer.policy = agent.account.policy = Policy(per_tx_max=500, per_window_max=100, window_secs=3600)
    tool = FoliantPaymentTool(agent=agent, http=http, default_deposit=100, prefer_pool=False)
    for _ in range(34):
        tool.invoke({"url": "http://api/infer", "body": "warm"})  # 33 calls at 3 drain the 100-unit channel; the 34th finds no headroom
    assert FoliantBudgetMiddleware(agent).headroom() == 1
    mw = FoliantBudgetMiddleware(agent, min_headroom=3)
    graph = create_agent(_model_that_calls(tool.name, 1), tools=[tool], middleware=[mw])
    result = graph.invoke({"messages": [HumanMessage("go")]})
    tm = [m for m in result["messages"] if isinstance(m, ToolMessage)][0]
    assert tm.status == "error" and tm.content.startswith("BUDGET REFUSED")
    with pytest.raises(BudgetExhaustedError):
        create_agent(_model_that_calls(tool.name, 1), tools=[tool],
                     middleware=[FoliantBudgetMiddleware(agent, min_headroom=3, exit_behavior="error")]).invoke({"messages": [HumanMessage("go")]})


def test_middleware_ignores_other_tools(world):
    L, http, agent, _ = world
    agent.signer.policy = agent.account.policy = Policy(per_tx_max=1, per_window_max=1, window_secs=3600, expiry=L.now)
    mw = FoliantBudgetMiddleware(agent)  # budget is dead

    def echo(text: str) -> str:
        """Echo."""
        return text

    model = _ToolCallingFake(responses=[AIMessage(content="", tool_calls=[{"name": "echo", "id": "e1", "args": {"text": "hi"}}]),
                                        AIMessage(content="done")])
    result = create_agent(model, tools=[echo], middleware=[mw]).invoke({"messages": [HumanMessage("go")]})
    tm = [m for m in result["messages"] if isinstance(m, ToolMessage)][0]
    assert tm.content == "hi"


def test_crew_bounds_workers_and_the_crew(world):
    L, http, supervisor, gate = world
    supervisor.signer.policy = supervisor.account.policy = Policy(per_tx_max=500, per_window_max=150, window_secs=3600)
    crew = FoliantCrew(supervisor, http, default_deposit=50, prefer_pool=False)
    wp = Policy(per_tx_max=100, per_window_max=100, window_secs=3600)
    workers = [crew.worker(n, wp, fund=1_000) for n in ("research", "analysis", "writer")]
    refusals = 0
    for _ in range(40):
        for w in workers:
            if w.tool.invoke({"url": "http://api/infer", "body": "t"}).startswith("BUDGET REFUSED"):
                refusals += 1
    c = crew.committed()
    assert c["crew"] == 150 and sum(c[w.name] for w in workers) == 150
    assert all(c[w.name] <= 100 for w in workers) and refusals > 0
    got = crew.revoke("writer")
    assert got > 0
    assert workers[2].tool.invoke({"url": "http://api/infer", "body": "t"}).startswith("BUDGET REFUSED")
    assert workers[2].middleware.headroom() == 0
    assert gate.settle() > 0
    assert L.total_supply(ASSET) == 1_000 + 10_000
