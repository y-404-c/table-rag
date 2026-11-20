# 表格文档 RAG Agent：评测报告

生成时间：2026-11-21 00:42:36

```json
{
  "model": "qwen4-max",
  "embedding": "text-embedding-v3",
  "rerank": "gte-rerank-v2",
  "status": "evals_not_run_on_clean_test",
  "split_seed": 7,
  "note": "numbers pending; do not cite V1 seed-42 figures"
}
```

## 状态说明：评测尚未在干净 test 集上运行

本报告对应的代码已修复用户指出的 9 个验收缺口（commit `121403d`），但**评测数字尚未重跑**。原因：

1. 原 seed-42 的 test 集已被反复看结果调参污染，现已用 **seed 7 重新冻结干净 test 集**
   （dev 200 / test 300 / corpus_pool 1979，见 `data/splits.json`）；
2. 剩余 token 额度约 0.12M，不足以跑完整评测（干净 test + 开放召回 + 消融 + 诚实负例约需 1–2M）；
3. 按用户决定**暂不消耗额度重跑**，留待补充 key 后一次性跑出最终数字——test 只跑一次，不看结果调参。

> ⚠️ 本文件之前显示的 V1 数字（15 题、seed 42、污染集）**已失效，不再作为验收依据**。

### 已完成的代码修复（不耗 token，已落地 + 单测覆盖）

- **cell 级出处**：`eval/qa_metrics.py` 新增严格 `cell_citation_hit`（须 `table_id`+`row`+`col` 齐全且该单元格值命中 gold），与文本级 `citation_hit` 分开报告；`agent/graph.py` 非计算路径（含多轮指代）确定性地在检索证据行里扫描匹配答案的单元格，补全 `row+col`，让 `$392/$506` 这类数值答案能溯源到具体单元格。
- **防 test 污染**：断点签名加入 `_code_version()`（关键源文件内容哈希），改 prompt/代码即签名失效，杜绝复用改前结果；干净 test 集 seed 7；旧 `*_runs.jsonl` 已删除。
- **开放召回**：新增 `run_retrieval_eval`，不传 `doc_id` 在语料库（pool + test）上召回，算 `doc/cell recall@5/@20`——真正检验双路召回 + RRF + CrossEncoder 精排。
- **诚实负例**：新增 `in_corpus_no_answer`（pool 文档都在库里但不含答案，开放检索）；报告正文明确标注 scoped 协议的 1.0 为**送分协议，不代表真实拒答率**。
- **消融对照**：新增 `run_ablation`（full / no_rerank / no_bm25 / no_vector 对照表，W2 验收项）。
- **缓存口径**：`MetricsMixin` 加 `threading.Lock` 修并发 `+=` 竞态；每批评测前 `cache.reset()`，只报本批次 embedding 命中，报告标注口径。

### 一键复现（补充 key 后运行，test 只跑一次）

```powershell
$env:PYTHONPATH = "e:\resume\table-rag\src"
cd e:\resume\table-rag
python -m tablerag.cli eval run --qa-limit 15 --repeats 3 --workers 4 --parse-limit 30 --refusal-limit 50 --demo-limit 3 --honest-refusal --retrieval --ablation --corpus-size 200 --tag final
```

单独子命令：`tr eval qa --limit 15 --repeats 3`、`tr eval retrieval --limit 50 --corpus-size 500`、`tr eval ablation --limit 15 --repeats 3`、`tr eval refusal --protocol in_corpus --limit 50`。

补充 key 方式：在 `.env` 里把新 key 追加到 `TR_LLM__API_KEYS`（逗号分隔），KeyPool 会在额度耗尽时自动轮换。


## 局限与风险披露（必须随数字一起看）

1. **数据是英文财报**：TAT-QA 为英文数据集，中文场景说服力有限。没有为了好看自造中文评测集。
2. **渲染-解析同源**：PDF 由 reportlab 按固定 seed 程序生成、由 pdfplumber 解析，测的是
   "解析管线能否还原被程序渲染的表格"，不等于真实扫描件/复杂排版的解析能力。渲染与解析
   刻意使用不同工具，并随机化空白/合并单元格/跨页/双栏，但**无法消除循环论证**。
3. **代表性**：TAT-QA 表格规模偏小（多为 5~10 行），结论不外推到大型多页财报表。
4. **出处命中口径自定**：见 eval/qa_metrics.py 的 `citation_hit`，与官方指标不是同一口径。
5. **样本规模**：受模型 token 额度限制，本次为 test 集前 N 个文档/问题的小样本（N 见每节标题下），
   不是 300 个 test 文档的全量结论；同一条命令可扩样重跑，断点续跑不会重复花钱。
6. **本文件中的每一个数字都来自 `tr eval ...` 的输出**；未运行的项显示为"未运行"。
