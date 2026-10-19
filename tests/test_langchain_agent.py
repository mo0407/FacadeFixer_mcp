"""Verify create_agent's model/tool loop without sending a remote LLM request."""
import asyncio
import json
from pathlib import Path
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from facade_mcp.orchestration import run_agent
from facade_mcp.storage import ArtifactStore


class ScriptedModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-tool-loop-test"

    def bind_tools(self,tools,**kwargs):
        assert any(tool.name=="survey_validate_site" for tool in tools)
        return self

    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        if any(isinstance(message,ToolMessage) for message in messages):
            response = AIMessage(content="offline validation completed")
        else:
            response = AIMessage(content="",tool_calls=[{"id":"call-site","name":"survey_validate_site","args":{
                "site":{"building_id":"B1","longitude":120.15,"latitude":30.28,"building_height_m":20}}}])
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_langchain_agent_really_calls_mcp_tool(tmp_path,monkeypatch):
    monkeypatch.setenv("FACADE_LLM_API_KEY","offline-test-placeholder")
    monkeypatch.setattr("langchain_openai.ChatOpenAI",lambda **kwargs:ScriptedModel())
    store = ArtifactStore(tmp_path/"workspace",[tmp_path])
    result = asyncio.run(run_agent("验证建筑坐标",store,model="offline-scripted-test"))
    assert result["final_answer"]=="offline validation completed"
    messages = json.loads(Path(result["messages"]["path"]).read_text(encoding="utf-8"))
    tool_messages = [message for message in messages if message["type"]=="tool"]
    assert len(tool_messages)==1
    assert tool_messages[0]["name"]=="survey_validate_site"
    content = tool_messages[0]["content"]
    if isinstance(content,list):
        content = next(block["text"] for block in content if block.get("type")=="text")
    assert json.loads(content)["valid"] is True
