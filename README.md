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
- **评测是长跑，不是一次性脚本**。每个「问题 × 重复」的原始结果立即追加 JSONL，中断后重跑自动跳过；
  key 额度耗尽时按 `TR_LLM__API_KEYS` 顺序自动轮换，换 key 后从断点继续，不重复花钱。

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
pytest                                                       # 42 个单测，全部不依赖外部 API
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
| 系统 | 延迟 / token / 缓存命中率 | 每次问答记录；报告给出 P50 / P95、总 token 与缓存命中率 |
| 系统 | 重复实验稳定性 | 同一题重复 3 次，报告 EM/F1 的题内标准差 |

### 4.3 当前结果

完整报告见 [reports/report.md](reports/report.md)（由脚本生成，数字来自 `reports/*.json`，未手填）。

**评测配置**：模型 `qwen4-max`（关闭思考链，`temperature=0`），embedding `text-embedding-v3`，
rerank `gte-rerank-v2`；解析 30 个 test 文档；问答 15 个 test 文档 × 每题 1 问 × 3 次重复；
负例 50；HyDE 对照 15 题 × 3 次；多轮 3 段 × 3 轮。

| 层 | 设置 | 结果 |
|---|---|---|
| 解析（30 文档） | 结构化 | content 0.9546 / index 0.9546 / 形状一致 1.0000 |
| 解析（30 文档） | 拍平基线 | content 0.9570 / index 0.0000 |
| 问答（15 题 ×3） | 结构化 | EM 0.7333 / F1 0.8964 / 出处命中 0.7143 / 拒答 0.0667 |
| 问答（15 题 ×3） | 拍平基线 | EM 0.7333 / F1 0.8972 / 出处命中 0.3333 / 拒答 0.0667 |
| 系统（结构化） | P50 / P95 / token | 10178 ms / 13924 ms / 3927 token per run |
| 负例（50） | scoped_missing_document | 正确拒答 1.0000 / 硬编率 0.0000 |
| HyDE（15 题 ×3） | 开 vs 关 | EM 0.7333 → 0.7333；F1 0.8964 → 0.9012；token 3927 → 4381 |

**怎么读这些数字（不做美化）**：

- **结构化路径的价值体现在"可审计"**：拍平基线 EM/F1 与结构化几乎相同（这批题以段落问答为主），
  但出处命中率 0.3333 vs 0.7143——纯文本块没有行列坐标，引用大多无法定位回单元格。
  解析层 index accuracy 0.9546 vs 0 是同一件事的另一面。
- **HyDE 在这批题上没有收益**：EM 持平、F1 +0.0048（在噪声范围内），但每题多用约 450 token、
  P95 延迟从 13.9s 涨到 18.3s。如实记录为"此场景不值得默认开启"。
- **3 次重复的题内标准差为 0**（temperature=0）：说明输出稳定，但这也意味着"重复 3 次"检验的是
  调用稳定性而非采样多样性；这是刻意选择，避免为了方差而引入随机性。
- **负例协议是 scoped**：验证的是"检索为空时是否拒答"，不等于"文档不在语料池时是否硬编"；
  后者需要 `--rebuild` 重建索引，脚本已支持但本次未跑。
- **1 个问答被拒答**：gold 为段落型长答案，证据判断节点判定不足；属于漏答，保留在明细里。

> 本次是受 token 额度限制的小样本（15 题），不是 300 个 test 文档的全量结论。
> 扩样只需改大 `--qa-limit` 并重跑同一条命令，断点续跑会复用已有结果。

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
4. **延迟**：单题 3~5 次串行 LLM 调用；本机实测 P50 约 10 s、P95 约 14 s（qwen4-max，关闭思考链）。
   上线需要小模型路由 / 答案缓存 / 检索并发。
5. **指标口径**：出处命中率与 EM/F1 均为自实现，与 TAT-QA 官方脚本不是逐行一致。
6. **样本规模**：15 题 ×3 次的小样本，不是 300 文档全量；置信区间宽，结论只用于验证链路与相对趋势。
7. **多轮的第 3 轮**：3 段演示中 2 段能正确做"那 2018 年呢"的追问；但"两个值的差"在
   `table_calc` 定位不到表格单元格时会拒答（2/3 段），这是计算工具只认表结构的真实缺口。
8. **思考链默认关闭**：qwen4-max 默认开思考链，会吃 completion token 且偶发空 JSON；
   报告中的数字均为 `enable_thinking=false` 下取得，与开启思考链的结果不可直接比较。

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

tests/                   # 42 个单测：解析回环/检索/指标/RRF/工具/缓存/拒答闸门/key 轮换（不依赖外部 API）
data/splits.json         # 冻结划分（唯一允许提交的数据文件）
reports/                 # 评测 JSON/CSV 与 report.md
var/                     # 索引、缓存、trace、checkpoint（gitignore）
```
