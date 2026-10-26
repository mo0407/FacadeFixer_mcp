# 轻量网页端 · 第一版

基于蓝白 V2 对话界面实现，旧 UI 与 PRD 不覆盖。本版本使用本地 Python 服务 + 原生 HTML/CSS/JavaScript，不需要 Node 构建。图片检测、Mask 与报告数据来自现有 MCP，页面不伪造识别结果。

## 启动

在仓库根目录运行 PowerShell：

```powershell
.\setup.ps1 -Web -SkipUpstream
Copy-Item .env.example .env  # 仅首次；已有配置不要覆盖
```

编辑 `.env`，至少配置：

```dotenv
FACADE_SOURCE_REPO=../项目代码/FacadeFixer-main
FACADE_SOURCE_PYTHON=../项目代码/FacadeFixer-main/.venv/Scripts/python.exe
FACADE_SOURCE_DEVICE=cpu
FACADE_WEB_WORKSPACE=./workspace/web
```

上述源仓库需要已安装对应模型依赖和权重。网页不会自动下载权重。MCP 服务环境和模型环境可以使用不同 Python。

```powershell
.\run_web.ps1
# 已有可用环境时：
.\run_web.ps1 -Python 'E:\ai\低空\facade_mcp_langchain\.venv\Scripts\python.exe'
```

打开 <http://127.0.0.1:8780>。终端保持运行，Ctrl+C 停止。也可 `pip install -e ".[web]"` 后运行 `facade-web --port 8780`。

## 使用流程

1. 新建报告，填写建筑名称、位置和巡检期次。
2. 添加图片、视频、建筑资料，输入生成要求并发送。
3. 系统抽帧 → 调用 `facade_models_detect` → `facade_models_segment` → `report_build_draft`。网页显示阶段与失败原因。
4. 报告生成后，直接点击标题、概述、缺陷描述和处理建议编辑；类别与立面也可修改。停止输入约 900ms 后保存版本。
5. 可以继续发送修改要求，例如 `标题改为A栋复查报告`、`第1条改为白华`、`第1条处理建议改为现场复核后再确定修复方案`、`概述改为本次为定期复查`。
6. 查看修改记录中的修改前后对比。恢复旧版会创建新版本，不删除原记录。
7. 添加新的素材后，点击“重新分析”重新检测并生成新版本。普通对话修改不重复运行检测和分割。
8. 导出 HTML（内嵌图片，可离线浏览/浏览器打印）或 JSON。窄屏使用“查看报告 / 返回对话”切换。

## 本地与在线助手

默认本地模式：按照固定报告结构组合真实检测证据，保留用户生成要求；不会任意理解复杂排版、自动推断立面或润色全文。明确的本地修改指令无需语言模型。

自由生成要求和自然语言文字改写需配置语言模型，并在“建筑信息与模型配置”中主动启用在线助手：

```dotenv
FACADE_LLM_MODEL=你的模型名称
FACADE_LLM_BASE_URL=你的兼容接口地址
FACADE_LLM_API_KEY=在本机填写
```

在线助手通过 LangChain `ChatOpenAI` 调用兼容接口。发送内容包括用户指令、报告文字（可能包括附件摘录）和选中文字；不发送原始图片。它只能修改现有报告的文字、类别和立面，不能添加检测实例或更换证据、面积。文字助手失败时保留真实检测草稿。未配置 API 时给出明确提示；本轮未使用付费模型验证。

## 输入输出与范围

| 输入 | 处理 | 输出 / 限制 |
|---|---|---|
| JPG / JPEG / PNG / WebP | 验证文件后检测 | 单文件 ≤200MB，图片 ≤1亿像素 |
| MP4 / MOV / AVI / MKV | OpenCV 抽帧 | 每视频 1–30 帧，间隔至少 1–300 秒；长视频扩大间隔覆盖全片，保留秒级来源 |
| TXT / MD | UTF-8 文本摘录 | 最多12,000字符，报告默认引用前3,000字符 |
| PDF | 前20页文本提取 | 不做OCR；扫描版需补充文字 |
| 建筑信息 | 用户填写 | 不自动解析视频 GPS、高度或判定立面 |
| 模型配置 | 检测器、分割器、框/点/文本提示 | 文本提示仅 SAM3；U-Net 仅裂缝、框裁剪 |
| 编辑结果 | 乐观并发检查、HTML白名单 | 报告版本、HTML与JSON导出 |

每任务最多120项素材/证据；每轮最多100张图片或视频帧。模型文件缺失、空文件、非法类型、冲突保存、重复运行等会显示错误。模型类别修改属于人工记录；保留原模型类别与原 Mask，不声称改完类别就重新分割。

当前是单机单用户工作区。没有账号系统、多人授权、3D重建、正式签署、自动专业诊断、Word/PDF生成器，也未复刻上游报告仓库的所有版式。当前报告是现有 Report MCP 结构的可编辑网页呈现。

## 数据与接口

```
workspace/web/
  web.sqlite3        # 任务、消息、报告、不可覆盖的版本快照
  uploads/           # 原素材；界面移除只移出列表，不清除磁盘原件
  frames/            # 视频帧及来源时间
  mcp/               # MCP原始报告、Mask和证据
  trace/             # 实际工具调用与返回数据（本地调试）
```

报告版本保存时间、修改原因、提示词、完整快照和 SHA-256 链。所有操作者标识为“本地用户”；这不构成多用户身份认证或不可篡改的安全审计。没有实现自动训练、数据清洗或授权反馈集导出。后续训练应另行筛选/审核修改记录。

服务只监听 `127.0.0.1`，校验 Host 与修改请求的 Origin。不要直接转发到公网。上传、轨迹、数据库和模型权重均被 Git 忽略。

| 接口 | 用途 |
|---|---|
| `GET/POST /api/tasks` | 列表 / 新建 |
| `GET /api/tasks/{id}` | 详情与运行状态 |
| `PUT /api/tasks/{id}/settings` | 建筑与模型配置 |
| `POST /api/tasks/{id}/uploads` | 原始二进制上传，`X-File-Name` 为编码文件名 |
| `GET/DELETE /api/tasks/{id}/assets/{asset_id}` | 读取 / 移出素材列表 |
| `POST /api/tasks/{id}/chat` | 异步生成或修改，返回202；带 `base_revision` |
| `PUT /api/tasks/{id}/report` | 自动保存，过期版本返回409 |
| `GET /api/tasks/{id}/versions` | 版本与摘要链 |
| `GET/POST /api/tasks/{id}/versions/{n}` | 查看 / 恢复为新版本 |
| `GET /api/tasks/{id}/export?format=json` | HTML默认，JSON可选 |

## 验证

`python -m pytest tests/test_web_app.py -q`：上传隔离与校验、富文本过滤、证据不可改、版本冲突、历史恢复、视频抽帧、导出、重启恢复、提示词修改及失败保留。

2026-10-25 实际联调：仓库示例图片 → YOLO12m → MobileSAM（框）→ Report MCP，生成1项锈渍候选，Mask像素面积41630；仅说明链路跑通，不是准确率评估。其余模型及在线语言模型不在这次真实推理验证范围内。
