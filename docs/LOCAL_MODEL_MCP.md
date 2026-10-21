# FacadeFixer 本地模型 MCP 接入

已将 `项目代码/FacadeFixer-main` 的检测和 Mask 能力加入现有 MCP 工程。当前统一配置有 **4 个服务、21 个工具**。模型运行在源仓库的 Python 环境中，MCP/LangChain 使用本工程环境；不需要启动旧 FastAPI 后端。

## 1. 服务与模型

| MCP 服务 | 能力 | HTTP 默认端口 |
|---|---|---|
| survey | 建筑资料、航线草案、素材、原有检测入口 | 8761 |
| model | 本地模型目录、检测、Mask、正负点精修、任务与资产 | 8764 |
| mask | 原有 OpenCV、已授权 API Mask、OpenCV 精修 | 8762 |
| report | 报告草稿、验证、HTML 导出 | 8763 |

| profile | 用途 | 实现与限制 |
|---|---|---|
| `yolo12m` | 缺陷检测 | 复用源 `model_runner` 的加载器，受控设置阈值与设备 |
| `yolo11m` | 缺陷检测 | 同上 |
| `rtdetr` | 缺陷检测 | 复用源 RT-DETR 加载器 |
| `faster_rcnn` | 缺陷检测 | 独立固定 Detectron2 配置和本地权重，缺依赖明确失败 |
| `ensemble` | 多模型合并 | 固定 YOLO12m、YOLO11m、RT-DETR，全部成功后合并 |
| `intersection` | 多模型共识 | 同上，至少两个不同模型支持同类同位置候选 |
| `mobile_sam` | bbox/正负点 Mask | 复用源 baseline 的本地 Ultralytics SAM 调用方式 |
| `sam3` | bbox/正负点 Mask | 本地 Ultralytics SAM3 路径；不依赖原生 SAM3 SDK 的 `triton` 导入路径 |
| `unet_crack` | 裂缝 Mask | 复用源 UNet16 架构和预处理，CPU 安全加载，阈值 0.2；仅 C 类（兼容旧 M） |

合并使用 `max(IoU, intersection/smaller-area) >= 0.5` 分组，同一模型只贡献一次，坐标和分数取贡献者均值。这是明确记录的适配实现，与旧后端的模型成员和投票流程有差别。模型分数保持模型原意，不当成准确率。

没有接入训练、GT 对照、批量评价脚本作为推理工具。`yolo11n-seg.pt` 是通用 COCO 分割权重，不注册为建筑缺陷模型。旧后端中会下载其他权重、自动付费选择候选、图像修复的流程也不在这些本地模型工具中。

## 2. 配置与启动

在本工程目录执行：

```powershell
$env:FACADE_SOURCE_REPO='E:\ai\低空\项目代码\FacadeFixer-main'
$env:FACADE_SOURCE_PYTHON='E:\ai\低空\项目代码\FacadeFixer-main\.venv\Scripts\python.exe'
$env:FACADE_SOURCE_DEVICE='cpu'
$env:FACADE_SOURCE_TIMEOUT_SECONDS='600'
.\.venv\Scripts\python.exe -m facade_mcp.cli models
.\.venv\Scripts\python.exe -m facade_mcp.cli mcp-config --output .\mcp.local.json
```

当前目录结构下可自动找到相邻的 `项目代码/FacadeFixer-main` 和它的 `.venv`；部署到其他位置时显式配置路径。`.env.example` 已补齐对应字段，现有 `.env` 密钥没有被改写。`models` 只探测解释器、依赖和权重文件存在性；`available_unverified` 不代表成功加载或通过效果评价。

MCP 客户端导入 `mcp.local.json`，模型可发现所有 21 个工具。服务端配置仅由部署者设置，工具参数不能改写 Python、权重路径、命令、API 地址或密钥。单独启动：

```powershell
.\.venv\Scripts\python.exe -m facade_mcp.model_server
.\.venv\Scripts\python.exe -m facade_mcp.model_server --transport http --port 8764
```

每个服务实例同时只运行一个模型子进程，超时或繁忙明确失败；不自动重试。多个客户端若分别启动多个服务实例，会各自占用资源。检测/分割是同步 MCP 调用，结果返回后可通过 job ID 重新查询状态与资产。

## 3. 六个新增工具的输入输出

| 工具 | 输入 | 输出 |
|---|---|---|
| `facade_models_list` | 无 | 9 个 profile、依赖/权重缺失、设备、运行时信息 |
| `facade_models_detect` | `image_paths[]`, `model_profile`, `threshold` | `bbox_class_only_v1`、每张图 bbox/类别/置信度、未知类别、原图哈希、检测清单资产 |
| `facade_models_segment` | `image_path`, `defects[]`, `model_profile` | `facade-mask/v1`、原尺寸实例/并集/类别 Mask、叠加图、像素面积、模型和代码哈希 |
| `facade_models_refine` | 原图、上一版 manifest、`instance_id`、`positive_points[]`、`negative_points[]`、SAM profile | 新版 Mask 清单、父版本/父资产哈希、点提示事件，其他实例保留 |
| `facade_models_get_job` | `job_id` | 持久化的 running/completed/failed 状态、结果或错误 |
| `facade_models_get_artifact` | `artifact_id` 或 `artifact:...` | 文件路径、角色、SHA-256、大小等元数据 |

`image_paths`、原图和 manifest 支持允许目录中的路径或当前 workspace 已注册的 `artifact:...`。`defects` 示例：

```json
[{"instance_id":"D-01","class_code":"C","bbox_xyxy":[120,65,245,395]}]
```

类别采用 C/P/S/T/R/D/V/L/E：C 合并线状裂缝与网裂，E 为独立白华类，L 为渗漏/潮湿。旧 M 自动归入 C；旧 L 仅在标签明确为白华时归入 E，保留原始来源。检测得到的 Bulging 等未知类单独保留，工作流把它们带入报告待处理条目，不删除也不伪造 Mask。U-Net 遇到非裂缝类别直接报错，应改选 SAM，或按类别分别调用。

所有坐标均为原图存储像素：左上角零起点，xyxy 的最大边界不包含；检测小数框以 floor(min)/ceil(max) 包围取整。保留 EXIF 信息但不旋转栅格，检测与 Mask 坐标一致。Mask 限制在被冻结的 bbox 内，空结果仍保留实例和复核标记。RLE 统一使用现有 F order，在完整 JSON 资产中保存，MCP 返回值移除大体量 RLE。

精修支持同一 bbox 内的正负点重新预测，不把上一版 Mask 作为模型 logits 输入。上一版保持不变，新版记录 parent、revision 和事件哈希来源；这不是画笔或多边形逐像素编辑。结果始终 `expert_validated=false`，无标定时 `physical_area_m2=null`。

## 4. LangGraph / LangChain

工作流选择源模型：

```json
{
  "detection":{"method":"source","model_profile":"yolo12m","threshold":0.25},
  "mask":{"method":"source","model_profile":"mobile_sam"}
}
```

完整配置见 `examples/source_models.workflow.json`，含现有真实照片样例（演示坐标不代表照片定位）：

```powershell
.\run_source_models.ps1
# 或
.\.venv\Scripts\python.exe -m facade_mcp.cli run --config .\examples\source_models.workflow.json
```

固定流程：坐标 → 航线草案 → 素材 → `facade_models_detect` → `facade_models_segment` → 原报告 MCP。可以分别选择 source 检测 + OpenCV Mask、原 annotations 检测 + source Mask。原有 `annotations/http/local_yolo` 与 `opencv/api` 入口继续可用。

LangChain `agent` 从同一配置动态发现新增工具，自主选择 profile；`run` 则由 JSON 决定调用次序。源码模型工具不使用付费 API；原 API Mask 继续使用它原有的显式授权流程。

## 5. 资产和可追溯性

每次调用创建独立 job：原图快照、worker 输入/输出、检测框或 Mask、叠加图、完整清单分开保存。Mask 同时保留模型原始未裁剪结果 `raw_model_mask` 和限制在 bbox 内的选用结果，以及裁除像素数。记录源文件哈希、适配代码哈希、实际权重 SHA-256、模型参数、设备与版本。推理子进程不继承 API 密钥，不加载旧后端 `.env.local`，阻断联网和自动安装/下载；缓存仅写入 job。

原图快照在推理后再次验哈希。精修检查父清单与原图一致、Mask 哈希、尺寸和二值性。错误、依赖缺失、超时均进入 failed；它们不代表没有缺陷。报告仍为现有 JSON/HTML 草稿格式。

## 6. 本机验证记录

开发环境已完成以下验证。`outputs/source_models_verification/verification.json` 等本地结果未随 Git 仓库发布；克隆后运行示例会在自己的 `workspace/jobs/` 中生成新结果：

- 真实 YOLO12m、YOLO11m、RT-DETR 推理成功。
- 4 个真实 stdio MCP 服务下，YOLO12m → MobileSAM → 报告链路成功。
- MobileSAM 正负点精修、U-Net 裂缝分割成功。
- Faster R-CNN 缺 `detectron2`，调用明确失败。
- SAM3 工具已接入；本次可用内存低于 8 GiB，跳过其 3.45 GB 权重的真实加载验证。
- 合并/共识算法有受控测试，尚未作为效果评价运行。

运行示例后，根据返回的报告 HTML 资产路径打开报告。运行成功不等于缺陷判断正确；开发验证没有 GT 效果评测或真实在线 LLM 调用。
