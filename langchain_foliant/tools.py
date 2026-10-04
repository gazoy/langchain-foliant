"""FoliantPaymentTool: an agent pays an x402 endpoint through a Foliant channel or pool."""
from __future__ import annotations

from typing import Any, Literal, Optional

import httpx
from foliant import Agent
from foliant.errors import PolicyViolation
from foliant.x402 import AgentHttpClient
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class PaidRequest(BaseModel):
    url: str = Field(description="Full URL of the x402 endpoint to call")
    method: Literal["GET", "POST"] = Field(default="POST", description="HTTP method")
    body: str = Field(default="", description="Request body for POST, as text")


class FoliantPaymentTool(BaseTool):
    """Call an x402 endpoint and pay for it through the agent's Foliant account.

    The tool answers a 402 by opening a channel to, or joining the pool of, the
    provider named in the response, then signs one off-chain update per call.
    The agent's signer refuses any update outside its spending policy, and the
    ledger refuses any deposit outside the policy of the account or of any
    ancestor in its crew. A refusal is returned to the model as text, not raised.
    """

    name: str = "foliant_pay"
    description: str = (
        "Call an HTTP API that charges per request over x402, paying from this agent's "
        "Foliant budget. Returns the response body and the provider-signed receipt id. "
        "If the budget does not allow the payment, returns a message starting 'BUDGET REFUSED'."
    )
    args_schema: type[BaseModel] = PaidRequest
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Agent
    http: httpx.Client
    default_deposit: int = 100
    prefer_pool: bool = True
    _client: Optional[AgentHttpClient] = None

    def _get_client(self) -> AgentHttpClient:
        if self._client is None:
            self._client = AgentHttpClient(self.agent, self.http, default_deposit=self.default_deposit,
                                           prefer_pool=self.prefer_pool)
        return self._client

    @property
    def receipts(self) -> list[dict]:
        """Provider-signed receipts for every paid call so far."""
        return self._get_client().receipts

    def _run(self, url: str, method: str = "POST", body: str = "",
             run_manager: Optional[CallbackManagerForToolRun] = None, **_: Any) -> str:
        client = self._get_client()
        try:
            r = client.request(method, url, content=body.encode() if body else None)
        except PolicyViolation as e:
            return f"BUDGET REFUSED: {e}"
        except ValueError as e:  # channel or pool deposit exhausted
            return f"BUDGET REFUSED: {e}"
        receipt = client.receipts[-1] if client.receipts else None
        # The provider signs an envelope: the receipt body, with updateId inside it, plus a signature.
        rid = (receipt.get("body", {}).get("updateId", "") or "")[:16] if receipt else "none"
        return f"{r.status_code}\n{r.text}\n[receipt {rid}]"
