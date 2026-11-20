# 表格文档 RAG Agent

面向"结构感优先"的表格文档问答系统：先用**结构化解析 + 双路混合召回 + 可追溯计算**，再用 Agent 编排，
最后用同一套冻结数据、同一套指标把每一层改进量化出来。

> 一句话差异：它把表格当**单元格坐标系**用，而不是把所有内容拍平成文本块再塞进向量库。

数据集：TAT-QA（英文财报问答，含 2D 表格 + 段落 + 推导标注）。本仓库用 train + dev 共 2479 个上下文
（14,883 个问题）冻结划分出 dev 200 / test 300 / corpus_pool 1979。

---

## 1. 这个项目解决什么

| 环节 | 问题 | 本项目的做法 |
|---|---|---|
| 解析 | 表格被拍成一行行文本后，"2020 年营收"和"2019 年营收"在向量里几乎一样 | pdfplumber 恢复二维网格，保留页/行/列；行内保留列名做检索文本 |
| 召回 | 纯向量对数字、专名不敏感；纯 BM25 对同义改写真不好使 | BM25 + 向量双路 → RRF 融合 → CrossEncoder 精排，三路均可用开关消融 |
| 问答 | LLM 会心算、会编数，答案对不了账 | 数值题必须调用受限的 `table_calc` 工具（pandas 执行）；拿不到工具结果就拒答 |
| 溯源 | 只给"来自第 3 段"无法核对 | 引用到 `doc_id + page + table_id + row + col`，可点开单个单元格 |
| 拒答 | 证据不足时仍强行作答 | 证据充分性判断节点；不足 → HyDE 重查（有上限）或直接拒答 |
| 评测 | 改了参数不知道有没有变好 | 解析层（Cell Content / Cell Index Accuracy）+ 问答层（EM/F1/出处命中/拒答率）全部脚本化 |

---

## 2. 架构

```
                    ┌──────────────────────────── FastAPI ────────────────────────────┐
                    │  POST /documents   POST /ask   POST /chat（多轮, thread_id）     │
                    └───────────────┬───────────────────────────────┬─────────────────┘
                                    │                               │
   PDF ──ingest.parser──► ParsedDoc(2D网格+段落)          ┌─────────▼─────────┐
        (pdfplumber)            │                         │  LangGraph Agent  │
                                ▼                         │                   │
                        ┌───────────────┐                 │  route            │
                        │  SQLite Store │                 │    ↓              │
                        │  units/缓存   │                 │  retrieve ←──┐    │
                        └───────┬───────┘                 │    ↓         │    │
                                │                         │  grade ──HyDE┘    │
                 ┌──────────────┴──────────────┐          │    ↓              │
                 ▼                             ▼          │  compute(pandas)  │
        BM25（rank-bm25）             向量（Chroma + API）  │    ↓              │
                 └──────────────┬──────────────┘          │  generate         │
                                ▼                         │  (JSON+引用校验)   │
                         RRF 融合 → gte-rerank 精排        └─────────┬─────────┘
                                                                     │
                                         Answer(answer + cell级citations + trace)
```

关键设计取舍：

- **检索单元是"表格行"不是"表格"**。列名 + 值拼成 `row3 | 2019: 8,657 | 2018: 8,244`，
  既保留数值原貌，又能在引用时精确到行。
- **计算与生成解耦**。LLM 只选操作和坐标，pandas 执行；算出的数值会回填进生成 prompt，
  模型漏写时由代码补上，避免"答案对了但没出处"。
- **条件边有上限**。证据不足 → HyDE 改写重查，最多 2 轮（`LoopGuard`），之后必须拒答。
- **缓存只加速不改语义**。embedding 按文本哈希缓存；Redis 不可用时显式降级为 SQLite 并打日志。

---

## 3. 快速启动

```bash
# 1. 依赖（Python 3.10+；本机已在 Python 3.13 上验证）
pip install -e ".[dev]"

# 2. 配置：复制 .env.example 为 .env，填百炼 key（用于 chat / embedding / rerank）
cp .env.example .env

# 3. 数据：下载 TAT-QA 原始 JSON（走 GitHub raw / jsDelivr 镜像）并冻结划分
tr dataset fetch --splits train,dev
tr dataset split            # dev 200 / test 300 / pool 1979，写入 data/splits.json 后不可覆盖

# 4. 建索引（示例：测试集前 20 个文档）
tr index build --split test --limit 20 --reset

# 5. 提问
tr ask "What is the average value for sales per square metre for both F19 and F18?" \
       --doc "tatqa:00addb09-bf31-4240-a351-bc4480a3cecc"

# 6. 多轮（Checkpointer 落 SQLite）
tr chat --doc "tatqa:00addb09-bf31-4240-a351-bc4480a3cecc" --thread demo

# 7. 服务
tr serve --port 8000
#    POST /documents (multipart pdf, ?fidelity=structured|flatten)
#    POST /ask       {"question": "...", "doc_id": "..."}
#    POST /chat      {"thread_id": "...", "question": "...", "doc_id": "..."}
```

### 评测

```bash
tr eval parse   --limit 30            # 解析层：结构化 vs 拍平基线（不调用 LLM）
tr eval qa      --limit 15 --repeats 3 --workers 4          # 结构化路径：3 次重复 + 并发
tr eval qa      --limit 15 --repeats 3 --workers 4 --flatten # 同一批问题的拍平基线
tr eval qa      --limit 15 --hyde                            # 打开 HyDE（其余条件不变）
tr eval qa      --limit 15 --no-rerank --no-bm25             # 消融
tr eval refusal --limit 50 --workers 4                       # 负例：该拒答的是否拒答
tr eval demo    --limit 3                                    # 3 段多轮对话，落 reports/multi_turn_demo.md
tr eval run --parse-limit 30 --qa-limit 15 --repeats 3 --workers 4 --hyde --refusal-limit 50
pytest                                                       # 41 个单测，全部不依赖外部 API
```

**断点续跑 / key 轮换**：每次「问题 × 重复」的结果会立即追加到
`reports/qa_<mode>_<模型>_<签名>_runs.jsonl`；中断后重跑同一条命令会跳过已完成项。
一个 key 的额度耗尽时，程序自动换 `.env` 里 `TR_LLM__API_KEYS` 的下一个 key 继续跑：

```dotenv
TR_LLM__API_KEY=sk-first
TR_LLM__API_KEYS=sk-first,sk-second,sk-third   # 逗号 / 分号 / 换行分隔
```

注意：`tr eval refusal --rebuild` 会清空并重建整个索引（真实移除负例文档），耗时与费用较高；
默认 `scoped_missing_document` 只验证"检索为空时是否拒答"，报告里会写明用了哪种协议。

---

## 4. 数据与指标

### 4.1 数据

- 来源：TAT-QA 官方 `dataset_raw/tatqa_dataset_train.json|dev.json`（2D 表格数组 + 段落 + `derivation`）。
- 划分：`data/splits.json` 由 `seed=42` 一次生成并冻结；test 只在报告阶段使用（调参看 dev）。
- 生成态 PDF：`data/rendered/`，由 reportlab 按固定 seed 渲染，扰动包括
  跨页断表（表头重复）、合并单元格（SPAN）、双栏排版（表格左栏 + 正文右栏）、随机单元格 padding。
  解析变了就重渲染，**不把渲染产物当原始数据**。

### 4.2 指标定义

| 层 | 指标 | 定义 |
|---|---|---|
| 解析 | Cell Content Accuracy | 表格文字内容正确率（不看位置，多重集匹配） |
| 解析 | Cell Index Accuracy | 文字同时落在正确行列的比例 |
| 解析 | 形状一致率 | 行列数与真值完全一致 |
| 问答 | EM / F1 | TAT-QA 口径：多标准答案取最优；EM 数值严格、F1 对数值给 0.1% 容差 |
| 问答 | 出处命中率 | 引用定位到的单元格文字包含标准答案，或数值误差 <0.1%（口径自定，见 `eval/qa_metrics.py`） |
| 问答 | 拒答率 | `refused=true` 的比例；负例集上越高越好 |
| 系统 | 延迟 / token / 缓存命中率 | 每次问答记录，写入 `var/traces/*.json` 与 `reports/qa_eval.json` |

### 4.3 当前结果

见 [reports/report.md](reports/report.md)（由脚本生成，数字来自 `reports/*.json`，未手填）。
最近一次小样本冒烟结果（`tr eval run --parse-limit 10 --qa-limit 4 --refusal-limit 4`，非全量、非最终结论）：

- 解析层 10 个文档：content / index accuracy 均约 0.98，形状一致率 1.0；拍平基线 content 约 0.98 但 index accuracy 恒为 0。
- 问答层结构化路径 4 个问题：EM / F1 / 出处命中率均为 1.0，平均延迟约 25 s，平均约 5.7k token（4~5 次串行 LLM 调用，未做流式与并发）。
- 拍平基线同一批 4 个问题：EM 0.5，说明丢结构后数值题会算错或拒答（含"one-time transition tax"题）。
- 负例拒答 4 题：全部正确拒答（硬编率 0，协议 `scoped_missing_document`）。
- 未达标的地方如实留着：拍平路径上"one-time transition tax"题检索到相邻段落，
  答成 `$868.4 million`（gold `$883.0 million`）且未被拦下，属于证据判断节点的漏判；
  同一题在结构化路径上答对。
- 结构化与拍平的明细分别落在 `reports/qa_eval.json` 与 `reports/qa_eval_flat.json`。

> 全量 300 test / 14k 问题的数字尚未跑：需要用户确认额度后再执行 `tr eval run`（当前脚本已支持）。

---

## 5. 与需求文档的偏差（重要）

| 需求 | 实现 | 原因 |
|---|---|---|
| BGE-M3 / BGE-Reranker 本地推理 | 改用百炼 `text-embedding-v3` + `gte-rerank-v2` API | 本机无 GPU、2 核 2G，本地模型需下载 2~3GB 且精排很慢；接口层已抽象，换回本地只需替换 `llm/embeddings.py`、`llm/rerank.py` |
| Redis 缓存与 Checkpointer | 抽象接口 + 默认 SQLite；Redis 实现保留，连接失败显式降级 | Windows 无原生 redis-server；避免"假装跑通" |
| LangFuse 观测 | 本地 JSON trace（`var/traces/`），记录节点耗时/token/检索分数 | 自托管 LangFuse 成本高，先保证能复盘 bad case |
| 真实财报扫描件 | 用 reportlab 程序渲染 TAT-QA 表格 | 避免人工标注成本；**循环论证风险已在报告里披露**，结论只对"可解析的文本型 PDF"负责 |
| 前端流式 | 未做 | 需求文档列为"有余力再说"；后端已按可流式的节点拆分 |
| TAT-QA 全量 16,552 题 | 使用 train+dev 的 14,883 题 + 冻结划分 | test 无 gold，避免把测试集当训练集调参 |

---

## 6. 局限与风险

1. **渲染-解析同源**：PDF 由程序生成、由程序解析，测的是"解析管线能否还原被程序渲染的表格"，
   不等于真实扫描件能力。双栏/断页/合并单元格只是有限扰动。
2. **英文数据集**：TAT-QA 是英文财报，中文表格结论需要另做中文评测集（本仓库未自造）。
3. **表格规模小**：TAT-QA 表格多 5~10 行，页面级结构恢复能力未覆盖大型多页报表。
4. **延迟**：单题 4~5 次串行 LLM 调用，约 30~60 s；上线需要并行化 / 小模型路由 / 缓存答案。
5. **指标口径**：出处命中率与 EM/F1 均为自实现，与 TAT-QA 官方脚本不是逐行一致。

---

## 7. 目录

```
src/tablerag/
├── config.py            # pydantic-settings，TR_ 前缀，.env 覆盖
├── service.py           # 装配层：Store / Cache / Retriever / Agent
├── cli.py               # tr dataset|render|index|ask|chat|eval|stats|serve
├── domain/              # ParsedDoc / TableGrid / Unit / Answer，文本与数值工具
├── ingest/              # 下载、冻结划分、扰动渲染、pdfplumber 解析、pypdf 拍平基线
├── retrieval/           # BM25、Chroma 向量、RRF、精排管线（逐层开关）
├── agent/               # LangGraph 图、prompt、pandas 工具、SQLite Checkpointer
├── eval/                # 解析/问答/拒答指标与报告生成
├── api/                 # FastAPI 三端点
├── storage/             # SQLite 仓储、缓存抽象（SQLite/Redis/内存）
└── observability/       # 本地 JSON trace

tests/                   # 30 个单测：文本/指标/RRF/工具/缓存/拒答闸门（不依赖外部 API）
data/splits.json         # 冻结划分（唯一允许提交的数据文件）
reports/                 # 评测 JSON/CSV 与 report.md
var/                     # 索引、缓存、trace、checkpoint（gitignore）
```
