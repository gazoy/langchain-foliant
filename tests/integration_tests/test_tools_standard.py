from typing import Type

from langchain_foliant import FoliantPaymentTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from tests.conftest import build_world


class TestFoliantPaymentToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[FoliantPaymentTool]:
        return FoliantPaymentTool

    @property
    def tool_constructor_params(self) -> dict:
        _, http, agent, _ = build_world()
        return {"agent": agent, "http": http}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"url": "http://api/infer", "method": "POST", "body": "hello"}
