# 仓库内容与本地配置

本仓库提供建筑巡检的 MCP 服务和 LangChain/LangGraph 编排代码。

## 包含

- `src/facade_mcp/`：4 个服务、21 个工具及隔离模型适配器。
- `tests/`：契约、安全边界、资产和真实 MCP 流程测试。
- `examples/`：工作流、标注格式和公开样例照片。
- `docs/`：本地模型接入与部署边界说明。
- 安装脚本、运行脚本、依赖锁定文件和 `.env.example`。

## 需要在本地准备

- `.env` 或环境变量中的模型地址、密钥、允许读取目录。
- `setup.ps1` 准备的 Mask/Report 上游代码。
- 使用本地模型时，需要独立的 `FacadeFixer-main` 源码、Python 模型环境和权重。
- `mcp.local.json`：安装后使用 `python -m facade_mcp.cli mcp-config --output mcp.local.json` 生成当前机器的配置。

虚拟环境、密钥、本机配置、权重、日志、运行结果和 UI 设计交付不进入版本控制。报告、原图快照、Mask、修改版本和调用记录会在运行时写入 `workspace/`。请按实际数据管理要求备份这些运行资产。

示例照片沿用开发时使用的上游公开样本，仅用于验证软件链路；示例坐标不代表其实际地理位置。示例结果不用于评估识别准确率。
