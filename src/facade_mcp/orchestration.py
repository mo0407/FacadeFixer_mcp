"""LangGraph stages over real MCP tools and an optional LangChain model agent."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END
from .storage import ArtifactStore, utc_now


def server_config(workspace: Path, allowed_roots: list[Path], mask_repo: Path | None = None) -> dict:
    env = dict(os.environ)
    env.update(FACADE_WORKSPACE=str(workspace.resolve()),FACADE_ALLOWED_INPUT_ROOTS=";".join(str(p.resolve()) for p in allowed_roots),
               PYTHONIOENCODING="utf-8",FASTMCP_SHOW_SERVER_BANNER="false",FASTMCP_CHECK_FOR_UPDATES="off")
    config_file = Path(os.environ.get("FACADE_CONFIG_FILE",str(Path.cwd()/".env"))).resolve()
    if config_file.is_file():
        env["FACADE_CONFIG_FILE"] = str(config_file)
    if mask_repo:
        env["FACADE_MASK_REPO"] = str(mask_repo.resolve())
    # Include source dir for editable/source checkouts and preserve any existing path.
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return {"mcpServers":{name:{"command":sys.executable,"args":["-m",module],"env":env}
            for name,module in {"survey":"facade_mcp.survey_server","mask":"facade_mcp.mask_server","report":"facade_mcp.report_server","model":"facade_mcp.model_server"}.items()}}


def unpack_tool_result(result: Any) -> dict:
    if isinstance(result,dict):
        return result
    if hasattr(result,"artifact") and isinstance(result.artifact,dict):
        return result.artifact
    content = getattr(result,"content",result)
    if isinstance(content,str):
        parsed = json.loads(content)
        if isinstance(parsed,dict):
            return parsed
    if isinstance(content,list):
        for block in content:
            text = block.get("text") if isinstance(block,dict) else getattr(block,"text",None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed,dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    raise ValueError("MCP tool did not return a structured JSON object")


class ToolGateway:
    def __init__(self,tools:list,trace_path:Path):
        self.tools = {tool.name:tool for tool in tools}
        self.trace_path = trace_path

    async def call(self,name:str,arguments:dict) -> dict:
        if name not in self.tools:
            raise ValueError(f"Missing MCP tool {name}; available: {sorted(self.tools)}")
        started = time.perf_counter()
        row = {"tool":name,"arguments":arguments,"started_at":utc_now()}
        try:
            output = unpack_tool_result(await self.tools[name].ainvoke(arguments))
            row.update(status="success",output=output)
            if output.get("status") == "failed":
                raise RuntimeError(f"{name} failed: {output.get('error')}")
            return output
        except Exception as error:
            row.update(status="failed",error=str(error))
            raise
        finally:
            row["elapsed_seconds"] = round(time.perf_counter()-started,4)
            with self.trace_path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")


class WorkflowState(TypedDict,total=False):
    request:dict
    validated_site:dict
    route:dict
    media:dict
    detection:dict
    segmentation:dict
    report:dict
    validation:dict
    rendered:dict


def build_workflow(gateway:ToolGateway):
    graph = StateGraph(WorkflowState)

    async def site(state):
        return {"validated_site":await gateway.call("survey_validate_site",{"site":state["request"]["site"]})}

    async def route(state):
        return {"route":await gateway.call("survey_plan_route",{"site":state["validated_site"]["site"],"options":state["request"].get("planning",{})})}

    async def media(state):
        request = state["request"]
        return {"media":await gateway.call("media_organize_images",{"image_paths":request["image_paths"],"building_id":request["site"]["building_id"],"capture_metadata":request.get("capture_metadata",{})})}

    async def detection(state):
        settings = state["request"].get("detection",{})
        # Detector matcher binds annotations by original content hash, not filename.
        paths = [row["image_path"] for row in state["media"]["images"]]
        method = settings.get("method","annotations")
        if method == "source":
            result = await gateway.call("facade_models_detect",{"image_paths":paths,"model_profile":settings.get("model_profile","yolo12m"),"threshold":settings.get("threshold",.25)})
        else:
            result = await gateway.call("defect_detect",{"image_paths":paths,"method":method,"annotation_path":settings.get("annotation_path"),"threshold":settings.get("threshold",.25)})
        archived = {row["sha256"]:row for row in state["media"]["images"]}
        for image in result.get("images",[]):
            digest = image.get("source_sha256") or image.get("sha256")
            if digest not in archived:
                raise RuntimeError("Detector input differs from archived original photo")
            image["image_path"] = archived[digest]["image_path"]
        return {"detection":result}

    async def masks(state):
        settings = state["request"].get("mask",{})
        results = []
        for image in state["detection"]["images"]:
            arguments = {"image_path":image["image_path"],"defects":image["defects"]}
            if settings.get("method","opencv") == "opencv":
                result = await gateway.call("mask_segment_opencv",arguments)
            elif settings["method"] == "source":
                result = await gateway.call("facade_models_segment",arguments|{"model_profile":settings.get("model_profile","mobile_sam")})
            elif settings["method"] == "api":
                if not state["request"].get("allow_paid_api",False):
                    raise ValueError("mask.method=api requires allow_paid_api=true in the workflow request")
                result = await gateway.call("mask_start_api_job",arguments|{"model_profile":settings.get("model_profile")})
                job_id = result["job_id"]
                deadline = time.monotonic()+float(settings.get("timeout_seconds",1800))
                while True:
                    status = await gateway.call("mask_get_job",{"job_id":job_id})
                    if status["status"] == "completed":
                        result = status["result"]
                        break
                    if status["status"] == "failed":
                        raise RuntimeError(str(status.get("error")))
                    if time.monotonic()>deadline:
                        raise TimeoutError("API mask job still running; use mask_get_job to retrieve it later")
                    await asyncio.sleep(2)
            else:
                raise ValueError("mask.method must be opencv, source or api")
            result["unmapped_defects"] = image.get("unmapped_defects",[])
            results.append(result)
        return {"segmentation":{"schema_version":"facade-mask-batch/v1","images":results}}

    async def report(state):
        project = dict(state["request"].get("project",{}))
        project.setdefault("project_id",state["request"]["site"]["building_id"])
        project.setdefault("building_id",state["request"]["site"]["building_id"])
        project.setdefault("name",state["request"]["site"].get("name") or state["request"]["site"]["building_id"])
        project["site"] = state["validated_site"]["site"]
        project["route_plan"] = state["route"]
        return {"report":await gateway.call("report_build_draft",{"project":project,"segmentation_result":state["segmentation"],"detection_result":state["detection"]})}

    async def validate(state):
        artifact = state["report"]["report_document"]
        return {"validation":await gateway.call("report_validate",{"document_path":artifact.get("uri") or artifact["path"]})}

    async def render(state):
        artifact = state["report"]["report_document"]
        return {"rendered":await gateway.call("report_render_html",{"document_path":artifact.get("uri") or artifact["path"]})}

    for name,node in [("site",site),("route",route),("media",media),("detection",detection),("masks",masks),("report",report),("validation",validate),("render",render)]:
        graph.add_node(name,node)
    sequence = ["site","route","media","detection","masks","report","validation","render"]
    graph.add_edge(START,sequence[0])
    for a,b in zip(sequence,sequence[1:]):
        graph.add_edge(a,b)
    graph.add_edge(sequence[-1],END)
    return graph.compile()


async def run_workflow(request:dict,store:ArtifactStore,mask_repo:Path|None=None) -> dict:
    from langchain.mcp import MCPAdapter
    from fastmcp.client.transports import MCPConfigTransport
    job_id,directory = store.create_job("workflow",request)
    trace = directory/"tool_calls.jsonl"
    try:
        transport = MCPConfigTransport(server_config(store.root,store.allowed_input_roots,mask_repo),name_as_prefix=False)
        async with MCPAdapter(transport) as adapter:
            tools = await adapter.list_tools()
            state = await build_workflow(ToolGateway(tools,trace)).ainvoke({"request":request})
        result = {"job_id":job_id,"status":"completed","orchestration":"langgraph_over_mcp","outputs":{k:v for k,v in state.items() if k!="request"}}
        store.write_json(directory/"workflow_result.json",result)
        result["manifest"] = store.artifact(directory/"workflow_result.json","workflow_result",job_id)
        result["trace"] = store.artifact(trace,"tool_call_trace",job_id)
        store.complete_job(job_id,result)
        return result
    except Exception as error:
        store.fail_job(job_id,{"message":str(error),"trace_path":str(trace)})
        raise


async def run_agent(prompt:str,store:ArtifactStore,mask_repo:Path|None=None,model:str|None=None) -> dict:
    from langchain.agents import create_agent
    from langchain.mcp import MCPAdapter
    from langchain_openai import ChatOpenAI
    from fastmcp.client.transports import MCPConfigTransport
    model_name = model or os.environ.get("FACADE_LLM_MODEL")
    if not model_name:
        raise ValueError("Set FACADE_LLM_MODEL or pass --model to select the tool-capable language model")
    key = os.environ.get("FACADE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set FACADE_LLM_API_KEY or OPENAI_API_KEY")
    kwargs = {"model":model_name,"api_key":key,"temperature":0,"timeout":120,"max_retries":0}
    if os.environ.get("FACADE_LLM_BASE_URL"):
        kwargs["base_url"] = os.environ["FACADE_LLM_BASE_URL"]
    llm = ChatOpenAI(**kwargs)
    job_id,directory = store.create_job("agent",{"prompt":prompt,"model":model_name})
    policy = """你是建筑外立面巡检流程助手，使用提供的MCP工具工作。先明确建筑坐标和高度，然后规划航线草案、整理已提供照片、获得缺陷框、生成mask、生成报告草稿。航线不连接真实飞行器，也不是已验证的可执行航线。没有照片时只生成规划并说明缺失。可先用facade_models_list查看本地模型和依赖是否就绪，按用户目标选择已配置的本地检测、分割或点提示精修工具；不能下载权重或编造可用能力。没有检测服务、可用本地检测模型或真实标注时停止识别阶段，不能编造bbox。使用每个工具真实返回的artifact URI/路径；不要猜测中间文件。API mask工具可能付费，只有用户明确要求API分割且服务配置启用时才使用；本地分割模型和OpenCV不需要付费API授权。模型结果始终待复核，不把模型输出标成正式量测、专家确认或脱敏已批准。检测置信度和分割模型评分不能当作准确率；没有真值不报告准确率。保留未知类别、空mask和失败实例。最终列出可打开的产物路径和未完成步骤。"""
    try:
        transport = MCPConfigTransport(server_config(store.root,store.allowed_input_roots,mask_repo),name_as_prefix=False)
        async with MCPAdapter(transport) as adapter:
            tools = await adapter.list_tools()
            agent = create_agent(llm,tools=tools,system_prompt=policy)
            response = await agent.ainvoke({"messages":[{"role":"user","content":prompt}]},config={"recursion_limit":80})
        messages = [message.model_dump(mode="json") for message in response["messages"]]
        store.write_json(directory/"messages.json",messages)
        result = {"job_id":job_id,"status":"completed","model":model_name,"final_answer":response["messages"][-1].content,
                  "messages":store.artifact(directory/"messages.json","agent_messages",job_id)}
        store.complete_job(job_id,result)
        return result
    except Exception as error:
        store.fail_job(job_id,str(error))
        raise
