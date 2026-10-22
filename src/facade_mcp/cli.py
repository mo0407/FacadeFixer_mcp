from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from .storage import ArtifactStore
from .demo import create_demo
from .orchestration import run_agent, run_workflow, server_config


def load_request(path:Path)->dict:
    path = path.resolve()
    request = json.loads(path.read_text(encoding="utf-8-sig"))
    def resolve(value):
        value = str(value)
        return value if value.startswith("artifact:") else str((path.parent/Path(value)).resolve())
    request["image_paths"] = [resolve(value) for value in request.get("image_paths",[])]
    annotation = request.get("detection",{}).get("annotation_path")
    if annotation:
        request["detection"]["annotation_path"] = resolve(annotation)
    return request


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv(Path(os.environ.get("FACADE_CONFIG_FILE",str(Path.cwd()/".env"))),override=False)
    parser = argparse.ArgumentParser(description="建筑巡检 MCP + LangChain 第一版")
    parser.add_argument("--workspace",type=Path,default=Path(os.environ.get("FACADE_WORKSPACE","workspace")))
    parser.add_argument("--allow-root",action="append",type=Path,default=[])
    parser.add_argument("--mask-repo",type=Path,default=Path(os.environ["FACADE_MASK_REPO"]) if os.getenv("FACADE_MASK_REPO") else None)
    sub = parser.add_subparsers(dest="command",required=True)
    run = sub.add_parser("run",help="通过 LangGraph 和 MCP 执行配置中的完整流程")
    run.add_argument("--config",type=Path,required=True)
    demo = sub.add_parser("demo",help="无付费调用：合成照片＋已给检测框＋真实OpenCV＋报告")
    demo.add_argument("--fixture-dir",type=Path,default=Path("examples/generated_demo"))
    agent = sub.add_parser("agent",help="LangChain模型自主调度MCP，需要自己的模型配置")
    agent.add_argument("--prompt",required=True)
    agent.add_argument("--model")
    config = sub.add_parser("mcp-config",help="输出四个stdio服务的客户端配置（survey、mask、report、model）")
    config.add_argument("--output",type=Path)
    sub.add_parser("models",help="检查本地模型配置、权重和依赖（不执行模型推理）")
    args = parser.parse_args()
    roots = [Path.cwd(),*[Path(value) for value in os.environ.get("FACADE_ALLOWED_INPUT_ROOTS","").split(";") if value.strip()],*args.allow_root]
    if args.command == "run":
        roots.append(args.config.resolve().parent)
    if args.command == "demo":
        roots.append(args.fixture_dir.resolve())
    store = ArtifactStore(args.workspace,roots)
    if args.command == "mcp-config":
        result = server_config(store.root,roots,args.mask_repo)
        # Config contains only explicit relevant env; never export process secrets.
        keep = {"FACADE_WORKSPACE","FACADE_ALLOWED_INPUT_ROOTS","FACADE_MASK_REPO","FACADE_CONFIG_FILE",
                "FACADE_REPORT_REPO","FACADE_MASK_MODELS_CONFIG","FACADE_DETECTOR_URL","FACADE_DETECTOR_MODEL",
                "FACADE_DETECTOR_PYTHON","FACADE_DETECTOR_WEIGHTS","FACADE_DETECTOR_DEVICE",
                "FACADE_SOURCE_REPO","FACADE_SOURCE_PYTHON","FACADE_SOURCE_DEVICE","FACADE_SOURCE_TIMEOUT_SECONDS",
                "FACADE_SELECTOR_URL","FACADE_SELECTOR_MODEL",
                "PYTHONPATH","PYTHONIOENCODING","FASTMCP_SHOW_SERVER_BANNER","FASTMCP_CHECK_FOR_UPDATES"}
        for server in result["mcpServers"].values():
            server["env"] = {key:value for key,value in server["env"].items() if key in keep}
        if args.output:
            store.write_json(args.output,result)
    elif args.command == "demo":
        result = asyncio.run(run_workflow(create_demo(args.fixture_dir),store,args.mask_repo))
    elif args.command == "run":
        result = asyncio.run(run_workflow(load_request(args.config),store,args.mask_repo))
    elif args.command == "models":
        from .model_service import ModelService
        result = ModelService(store).list_tools()
    else:
        result = asyncio.run(run_agent(args.prompt,store,args.mask_repo,args.model))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
