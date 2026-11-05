# 在另一台 Windows 电脑本地部署

目标：在新电脑打开 <http://127.0.0.1:8780/>，上传建筑照片，通过本机 YOLO12m 检测、MobileSAM 分割，生成可编辑报告。同时启动五个可供 MCP 客户端连接的服务。无需 GitHub Pages、GPU 或 API Key；首次安装依赖和下载上游代码需要联网。以下命令在 **Windows x64 PowerShell** 中执行。

## 1. 准备代码与权重

安装 Git for Windows 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，确认 `git --version`、`uv --version` 可用。使用 Python 3.12（可用 `uv python install 3.12` 安装）。选择任意可写的父目录，在其中运行：

```powershell
git clone https://github.com/mo0407/FacadeFixer_mcp.git
cd FacadeFixer_mcp
```

从原电脑上的 `项目代码/FacadeFixer-main` 复制模型所用的 **源码和权重**，放在克隆目录旁边。保留源码的相对目录结构，至少包含完整的 `backend/`、`model_runner/` 以及下列两份权重。不要复制旧电脑的虚拟环境、密钥配置、用户数据或运行结果。MCP 仓库不分发预训练权重，也不会在检测时下载它们。

```text
自选父目录/
  FacadeFixer_mcp/                        ← 在此运行后续命令
  FacadeFixer-main/
    backend/run_sam_bbox_baseline.py
    model_runner/adapters/detection/yolo.py
    model_weights/detection/yolo12m_building.pt
    model_weights/mobile_sam.pt
```

如需复现已验证的模型版本，可用 `Get-FileHash` 比对 SHA-256：

| 权重 | 本次验证使用的 SHA-256 |
|---|---|
| `yolo12m_building.pt` | `6761a5fd8e2669b49d6be5c2480e50c0450e731213eb4ba1c7e7b133dfa66749` |
| `mobile_sam.pt` | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |

## 2. 分别安装服务环境和模型环境

**网页和 MCP 环境：** 在刚克隆的 `FacadeFixer_mcp` 根目录执行。脚本创建本仓库的 `.venv`，按锁文件安装依赖和网页组件，并将指定版本的 Mask、Report 上游代码下载到 `vendor/`。不需要启动这两个上游仓库的网页。

```powershell
.\setup.ps1 -Web -Python 3.12
if ($LASTEXITCODE -ne 0) { throw '安装失败，请检查上面的错误' }
```

**模型环境：** 模型依赖单独安装在 `FacadeFixer-main/.venv`。以下是默认 YOLO12m + MobileSAM 的 CPU 安装方式；如已有可用的模型环境，可跳过安装，直接在下一步填写其 Python 路径。

```powershell
uv venv --python 3.12 ..\FacadeFixer-main\.venv
$modelPy = '..\FacadeFixer-main\.venv\Scripts\python.exe'
uv pip install --python $modelPy --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0' 'torchvision==0.28.0'
uv pip install --python $modelPy 'ultralytics==8.4.115' 'timm==1.0.28'
& $modelPy -c "import torch, torchvision, cv2, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__)"
```

每条命令成功后再继续。以上版本来自验证机器；若安装提示找不到对应轮子，请先检查 Python 为 64 位 3.12、网络可访问两个包源。GPU 或其他模型需要其对应的运行环境和权重；Faster R-CNN 还需要 Detectron2。视频工具导出 PDF 可另装 `uv pip install --python .\.venv\Scripts\python.exe 'reportlab>=4,<5'`，网页 HTML 报告不需要它。

## 3. 配置并启动

```powershell
if (-not (Test-Path .env)) { Copy-Item .\deploy\local.env.example .env }
.\.venv\Scripts\python.exe -m facade_mcp.cli models
.\run_backend.ps1 Start
.\run_backend.ps1 Status
```

[配置模板](../deploy/local.env.example)默认按上面的目录布局运行。若源码在其他位置，编辑 `.env` 中的 **`FACADE_SOURCE_REPO`**（包含 `backend/` 和 `model_runner/` 的目录）及 **`FACADE_SOURCE_PYTHON`**（模型环境的解释器）。Windows 绝对路径可写为 `D:/models/FacadeFixer-main`。不要把模型解释器传给 `run_backend.ps1 -Python`，该参数指定的是 **MCP 服务**解释器。已有配置不会自动覆盖；核对新模板中的目录后再启动。

在 `models` 输出中检查 `yolo12m`、`mobile_sam` 的依赖和权重；`available_unverified` 表示可以尝试推理，不能代替下一步的实图验收。`Status` 为 `running` 表示服务已监听。

## 4. 用一张图片验收

1. 打开 <http://127.0.0.1:8780/> → 新建报告 → 填写建筑名称。选择 YOLO12m、MobileSAM、bbox 提示，保持本地模式。
2. 上传仓库自带的 [facade.jpg](../examples/real_photo_sample/facade.jpg)，输入“请生成外墙巡检报告，并附上缺陷标注图片”。
3. 等待报告保存，打开缺陷页检查框位置图、局部特写、二值 Mask 和叠加图。可直接编辑报告，并从“导出”下载内嵌图片的 HTML。

原电脑实测：该样例得到 **1 处锈渍候选**，Mask 面积为 **41630 像素**。这是链路验收参考，不是模型准确率；更换权重或运行环境后结果可能不同。没有配置在线助手时，固定格式报告和明确的本地修改指令仍然可用。参见[网页操作与快捷标注](web-app.md)。

## 管理、连接与排错

```powershell
.\run_backend.ps1 Status
.\run_backend.ps1 Stop    # 等当前任务结束再停止
.\run_backend.ps1 Start
```

服务只监听本机，隐藏运行。电脑重启后手动执行 Start；报告、素材和修改版本保存在 `workspace/web/`，Stop/Start 不会清除数据。GitHub Pages 仍是静态演示，不能调用别人电脑上的 localhost。

| 本机客户端连接（Streamable HTTP） | 地址 |
|---|---|
| Survey：建筑与航线 | `http://127.0.0.1:8761/mcp` |
| Mask：分割与精修 | `http://127.0.0.1:8762/mcp` |
| Report：报告 | `http://127.0.0.1:8763/mcp` |
| Model：本地模型 | `http://127.0.0.1:8764/mcp` |
| Video：视频处理 | `http://127.0.0.1:8765/mcp` |

浏览器界面在 8780；上述五个地址是协议端点。网页自身按任务通过 stdio 调用 MCP。启动管理器另用 18780 防止重复启动。

| 现象 | 检查 |
|---|---|
| 安装或下载失败 | 确认 Git、uv、Python 3.12 可用，并能访问 GitHub、Python 包源和 CPU 轮子源。若 `vendor/` 已有上游仓库，确认是 `setup.ps1` 指定的版本。 |
| 启动失败或端口占用 | 用 Status 和 `.task_tmp/local-backend/supervisor.err.log` 检查原因，再读相应的 `<服务名>.log`。若单独运行过 `run_web.ps1`，先在原终端正常停止它。 |
| 页面可打开，但检测失败 | 重新运行 `facade_mcp.cli models`，检查模型解释器、源码和权重；修改 `.env` 后 Stop、Start。真实工具调用保存在 `workspace/web/trace/`。 |
