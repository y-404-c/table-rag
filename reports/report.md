# 表格文档 RAG Agent：评测报告

生成时间：2026-11-20 23:33:01

```json
{
  "model": "qwen4-max",
  "embedding": "text-embedding-v3",
  "rerank": "gte-rerank-v2",
  "repeats": 3,
  "workers": 4,
  "hyde_compare": true,
  "keypool": {
    "size": 1,
    "active_index": 0,
    "dead": 0,
    "rotations": 0
  },
  "total_qa_tokens": 543559
}
```

> 本报告由 `tr eval run` 生成；所有数字来自 reports/*.json，未手工填写。

## 1. 解析层（结构化路径 vs 拍平基线）

样本数：30 个文档；渲染扰动：跨页断表 / 合并单元格 / 双栏 / 随机空白。

| 路径 | Cell Content Accuracy | Cell Index Accuracy | 形状完全一致率 |
|---|---|---|---|
| 结构化（pdfplumber 表格恢复） | 0.9546 | 0.9546 | 1.0000 |
| 拍平基线（pypdf 纯文本） | 0.9570 | 0.0000 | n/a（无结构） |

差值：content -0.0024 / index 0.9546。

失败样例（前 5 个，按输入 order 而非挑最差）：

- 00addb09-bf31-4240-a351-bc4480a3cecc: gold [12, 5] vs pred [12, 5]，content=0.9649 index=0.9649，缺失样例 [(1, 3, 'CHANGE'), (1, 4, 'NORMALISED')]
- 02e79763-9b9e-499a-8be5-483dee8d681d: gold [6, 4] vs pred [6, 4]，content=0.8500 index=0.8500，缺失样例 [(5, 1, '—'), (5, 2, '—')]
- 0763f40c-4de1-417b-b3b0-7080ce64ba5b: gold [19, 3] vs pred [19, 3]，content=0.9804 index=0.9804，缺失样例 [(6, 2, '—')]
- 07a4ef02-b519-4377-bee3-43a5c301ccd9: gold [13, 4] vs pred [13, 4]，content=0.9773 index=0.9773，缺失样例 [(11, 3, '—')]
- 08011b49-022e-4e03-afa4-e897ca34e855: gold [16, 3] vs pred [16, 3]，content=0.9556 index=0.9556，缺失样例 [(3, 2, '—'), (5, 2, '—')]

## 2. 问答层（结构化路径）

样本数：15 个问题 × 3 次重复 = 45 次运行；并发 4；模型 `qwen4-max`。

| EM | EM 题内标准差 | F1 | F1 题内标准差 | 出处命中率 | 拒答率 | P50(ms) | P95(ms) | 平均 token | 总 token |
|---|---|---|---|---|---|---|---|---|---|
| 0.7333 | 0.0000 | 0.8964 | 0.0078 | 0.7143 | 0.0667 | 10178 | 13924 | 3927 | 176732 |

本批结果包含 30 次断点续跑复用的运行（同一签名、同一命令；新跑 15 次）。
缓存后端命中率：1.0000（hits=15, misses=0）
检索开关：{'bm25': True, 'vector': True, 'rerank': True}

掉点样例（前 5 个，按出现顺序；不含满分题）：

- Q: What are TSRs?
  gold=['performance shares that are earned, if at all, based upon the Company’s total shareholder return as compared to a group of peer companies over a three-year performance period.'] | pred="Total shareholder return awards — performance shares earned, if at all, based on the Company's total shareholder return compared to a group of peer companies over a three-year performance period, with payout ranging from 0% to 200%" | route=text_qa | refused=False | EM=0.0 | citation_hit=0.0
- Q: What event impacted the 2019 results?
  gold=['2019 results were impacted by Red Hat purchase accounting and acquisition-related activity.'] | pred='Red Hat purchase accounting and acquisition-related activity' | route=text_qa | refused=False | EM=0.0 | citation_hit=1.0
- Q: What is included in Client solutions?
  gold=['Client solutions often include IBM software and systems and other suppliers’ products if the client solution requires it.'] | pred="IBM software and systems and other suppliers' products if the client solution requires it" | route=text_qa | refused=False | EM=0.0 | citation_hit=0.0
- Q: What might the estimated potential impact being in the range of $75 million result in?
  gold=['a decrease in deferred tax assets and an increase in tax expense'] | pred='Insufficient evidence in the provided document to answer this question.' | route=text_qa | refused=True | EM=0.0 | citation_hit=0.0

## 3. 问答层（拍平基线）

样本数：15 个问题 × 3 次重复 = 45 次运行；并发 4；模型 `qwen4-max`。

| EM | EM 题内标准差 | F1 | F1 题内标准差 | 出处命中率 | 拒答率 | P50(ms) | P95(ms) | 平均 token | 总 token |
|---|---|---|---|---|---|---|---|---|---|
| 0.7333 | 0.0000 | 0.8972 | 0.0002 | 0.3333 | 0.0667 | 9027 | 13689 | 3770 | 169667 |

本批结果包含 30 次断点续跑复用的运行（同一签名、同一命令；新跑 15 次）。
缓存后端命中率：1.0000（hits=30, misses=0）
检索开关：{'bm25': True, 'vector': True, 'rerank': True}

掉点样例（前 5 个，按出现顺序；不含满分题）：

- Q: What are TSRs?
  gold=['performance shares that are earned, if at all, based upon the Company’s total shareholder return as compared to a group of peer companies over a three-year performance period.'] | pred="TSRs are performance shares that are earned, if at all, based upon the Company's total shareholder return as compared to a group of peer companies over a three-year performance period." | route=text_qa | refused=False | EM=0.0 | citation_hit=0.0
- Q: What event impacted the 2019 results?
  gold=['2019 results were impacted by Red Hat purchase accounting and acquisition-related activity.'] | pred='Red Hat purchase accounting and acquisition-related activity' | route=text_qa | refused=False | EM=0.0 | citation_hit=0.0
- Q: What is included in Client solutions?
  gold=['Client solutions often include IBM software and systems and other suppliers’ products if the client solution requires it.'] | pred="IBM software and systems and other suppliers' products if the client solution requires it" | route=text_qa | refused=False | EM=0.0 | citation_hit=0.0
- Q: What was the increase in net sales between 2018 and 2019?
  gold=['290.9'] | pred='Cannot produce a reliable numeric answer: the calculation tool could not locate the required cells.' | route=table_calc | refused=True | EM=0.0 | citation_hit=0.0

## 4. 负例拒答

负例数：50。协议：scoped_missing_document（提问指向一个不存在的文档，检验「无证据时是否拒答」；文档仍在语料池中）

| 正确拒答率 | 硬编率（应当拒答却作答） |
|---|---|
| 1.0000 | 0.0000 |

## 5. HyDE 开关对照（同一批问题、其余条件不变）

样本数：HyDE 关 15 题 × 3 次；HyDE 开 15 题 × 3 次；模型 `qwen4-max`。

| 条件 | EM | F1 | 出处命中率 | 拒答率 | P50(ms) | P95(ms) |
|---|---|---|---|---|---|---|
| HyDE 关 | 0.7333 | 0.8964 | 0.7143 | 0.0667 | 10178 | 13924 |
| HyDE 开 | 0.7333 | 0.9012 | 0.7143 | 0.0667 | 10201 | 18299 |

EM 差值：+0.0000 → 持平。

## 6. 多轮对话演示（Checkpointer 真实续接）

共 3 段对话；每段第 2 轮是「那…呢」式追问，第 3 轮需要复用前两轮的数值做差。

### what is the total compensation expense recorded by the company for the year ende（2018 vs 2019）

文档：`tatqa:0633bda8-10dd-4f06-b486-de35b19e672b`，thread：`demo-0633bda8-10dd-4f06-b486-de35b19e672b-20261120233127`

- **Q1**：What is the total compensation expense recorded by the Company for the year ended December 31, 2019?
  - A：$392
  - route=table_lookup | refused=False | 出处：p1 None rNone cNone
- **Q2**：And what about 2018?
  - A：$506
  - route=table_lookup | refused=False | 出处：p1 None rNone cNone
- **Q3**：What is the difference between the 2019 value and the 2018 value?
  - A：Cannot produce a reliable numeric answer: the calculation tool could not locate the required cells.
  - route=table_calc | refused=True | 出处：无

### how much was the foreign credit carryforwards before consideration of valuation （2018 vs 2019）

文档：`tatqa:08011b49-022e-4e03-afa4-e897ca34e855`，thread：`demo-08011b49-022e-4e03-afa4-e897ca34e855-20261120233127`

- **Q1**：How much was the foreign credit carryforwards before consideration of valuation allowance as of December 31, 2019?
  - A：$76.8 million
  - route=table_lookup | refused=False | 出处：p1 None rNone cNone
- **Q2**：And what about 2018?
  - A：$68.8 million
  - route=table_lookup | refused=False | 出处：p1 None rNone cNone
- **Q3**：What is the difference between the 2019 value and the 2018 value?
  - A：Insufficient evidence in the provided document to answer this question.
  - route=table_calc | refused=True | 出处：无

### how much is the research and development expenses for the year ended {y}?（2017 vs 2019）

文档：`tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656`，thread：`demo-0ff0244d-7a27-4650-94ab-9cd26268e656-20261120233127`

- **Q1**：How much is the research and development expenses for the year ended 2019?
  - A：$355,015 thousand
  - route=table_lookup | refused=False | 出处：p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 c1, p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 cNone, p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r2 cNone
- **Q2**：And what about 2017?
  - A：$135,997 thousand
  - route=table_lookup | refused=False | 出处：p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 c3, p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 cNone, p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r2 cNone
- **Q3**：What is the difference between the 2019 value and the 2017 value?
  - A：$219,018 thousand (2017 - Years ended December 31, / 2019 = -219,018)
  - route=table_calc | refused=False | 出处：p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 c1, p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 c3, p1 tatqa:0ff0244d-7a27-4650-94ab-9cd26268e656:t0 r3 cNone


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
