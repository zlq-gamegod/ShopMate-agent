# 闲鱼售后智能体 · ShopMate

基于 LangChain 1.x + LangGraph 的二手交易售后 Agent。规则问答、订单/物流/退款/纠纷查询、退款金额试算、越权转人工，全链路可评测。

场景是闲鱼：C2C 交易，平台只做居间，**不适用七天无理由退货**，纠纷靠客服介入与大众评审裁决。对话对象可能是买家也可能是卖家，同一笔争议两边问的是相反的问题——这是这个项目比商城客服多出来的一层。

## 评测结果

同一套代码，两组数据，`temperature=0`，每条跑 3 次取多数（`--repeat 3`）。

| 数据集 | 样本 | 通过率 | 检索执行率 | 未溯源数字 | 平均轮次 |
| --- | --- | --- | --- | --- | --- |
| 调优集 `eval_set.jsonl` | 45 | **97.78%**（44/45） | 100% | 0% | 1.47 |
| 留出集 `eval_set_holdout.jsonl` | 20 | **95.00%**（19/20） | 90.91%* | 0% | 1.45 |

\* 留出集脚本原始输出为 72.73%。其中 2 条（“还要等多久”“能退多少钱”）被标注成 `other`，而调优集里同类问法标的是 `policy`，属标注口径不一致，非真实缺陷，按统一口径重算为 10/11 = 90.91%。

留出集是写完 45 条之后另出的 20 条新题，**不参与任何调优**，只跑一次。两者差 2.78pp，说明没有严重过拟合；留出集唯一失败的一条（问“部分退款按什么比例裁定”，模型答“不按固定比例”，实际规则是 10%–30%）暴露的是关键词路由的召回上限——检索本身能召回正确条款，是路由没触发。详见 [`eval/BADCASES.md`](eval/BADCASES.md)。

优化轨迹（每轮只改一个变量）：

| 版本 | 改动 | 通过率 |
| --- | --- | --- |
| v1 | 基线 | 64.44% |
| v2 | 补路由关键词（`lexicon.py`） | 77.78% |
| v3 | 提示词 3 条 → 6 条 | 86.67% |
| v4 | 铁律：时效必须报签收天数 | 91.11% |
| v5 | 退款试算工具契约（允许按描述推断类型） | 95.56% |
| v6–v8 | 状态词原样引用、否定须给替代路径、补资金状态映射 | **97.78%** |

> v1–v5 为单次跑、`temperature=0.7`；v6 起改为 `temperature=0` + 3 次多数。qwen-plus 即使温度设 0 也有 ±1~2 条的抖动，单次跑的数字不足以归因，所以后三版统一了评测口径。

## 架构

```
用户消息
  │
  ▼
route_node ── classify() 关键词分三类
  │
  ├─ boundary（投诉/报警/起诉/转人工）──▶ handoff_node ──▶ 建工单，直接结束
  │
  ├─ policy（规则/时效/退款政策）──▶ retrieve_node ── 混合检索
  │                                      │
  │                                      ├─ 检索为空 ──▶ handoff_node（不硬答）
  │                                      └─ 有结果 ──▶ agent_node
  │
  └─ other（订单/物流/闲聊）──────────────────────▶ agent_node

agent_node ⇄ tools（ToolNode，10 个工具）
  │
  └─ 无工具调用 ──▶ 输出答案 / END
```

三层职责分离：

- `agent/state.py` —— 图共用的状态（`messages` / `route` / `retrieved` / `order_id` / `escalated`），`messages` 挂 `add_messages` reducer，由 checkpointer 按 `thread_id` 恢复历史
- `agent/nodes.py` —— 四个节点的业务逻辑与 system prompt
- `agent/graph.py` —— 只做装配：节点、条件边、`compile(checkpointer=InMemorySaver())`

## 三个关键设计决策

**1. 路由前置，而不是让 ReAct 自己决定要不要查库。**
纯 ReAct 下，规则类问题经常一轮就“凭记忆”作答，条款、天数、票数全凭想象。把「该不该检索」从模型手里拿出来，放到一个确定性的前置节点，检索执行率从 60.87% 提到 100%。代价是关键词覆盖有限——这是已知天花板，见上面的留出集结论。

**2. 订单事实走工具，不走 RAG。**
订单号是精确主键，BM25 和向量算的都是模糊相似度；400 字切块还会把一个订单的 10 个字段切散到不同 chunk，检索回来只有半张表。所以 `orders.json` 由 `@tool` 直接查字典，向量库里只有规则文档。**“事实要查，规则才要检索”** 是这套系统的分界线。

**3. 工具宁可多而窄，不要少而宽。**
10 个工具各自 docstring 写清「什么时候用 / 什么时候禁用 / 用户没给订单号先问别猜」，而不是合并成一个带 `aspect` 枚举的大工具。模型靠工具名二选一比靠参数值判断意图准确得多。实测把 `query_orders` 和 `query_logistics` 的 docstring 写成一样时，后者几乎不会被调用。

## 工具清单

| 工具 | 作用 |
| --- | --- |
| `query_orders` | 订单事实：状态、金额、商品、地址、卖家类型、签收时间 |
| `query_logistics` | 物流：快递公司、运单号、更新时间 |
| `get_after_sale_window` | 售后时效：签收天数、剩余天数、是否超期（**由代码算，不让模型减日期**） |
| `query_refund_status` | 退款单：状态、类型、金额、当前环节、截止时间 |
| `calc_refund_amount` | 退款试算：逐项列扣减/加项、运费承担方、最终金额、资金状态 |
| `query_guarantee` | 担保资金：钱在平台还是已放款、能否直接退 |
| `get_dispute_status` | 纠纷：阶段、举证截止、票数、下一步 |
| `search_knowledge` | 规则库混合检索（BM25 + 向量） |
| `get_current_time` | 当前时间 |
| `transfer_to_human` | 转人工，返回工单号 |

## 快速开始

```bash
# 1. 依赖（项目用 uv 建的 venv）
uv venv venv
uv pip install --python venv/Scripts/python.exe -r requirements.txt
# 或者：.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 环境变量
cp .env.example .env      # 填入自己的 DASHSCOPE_API_KEY
```

`.env`：

```ini
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxx          # 可选，不填也能跑
LANGSMITH_PROJECT=ShopMate
MODEL_TEMPERATURE=0.7               # 评测时设 0
```

Windows 下**始终用 `.\venv\Scripts\python.exe`**，不要用裸 `python`——PATH 上的 `python` 往往不是这个 venv，会出现“装完包却 ModuleNotFoundError”。

## 三种运行方式

**Web 界面（推荐，能看到完整执行过程）**

```bash
.\venv\Scripts\python.exe server.py      # http://127.0.0.1:8000
```

`server.py` 把图的执行过程（路由 → 检索 → 工具调用 → token）通过 SSE 推给浏览器，每条回复下方可展开看这一步到底做了什么；左侧切换会话与订单，点订单号直接带进输入框。改端口用 `SHOPMATE_PORT=8001`，局域网访问用 `SHOPMATE_HOST=0.0.0.0`。

**命令行多轮对话**

```bash
.\venv\Scripts\python.exe runner.py
```

**单次调用（评测/脚本用）**

```python
from runner import run_once

r = run_once("ORD202610180001 到哪了", thread_id="demo")
print(r["answer"], r["route"], r["tool_calls"], r["turns"])
```

`runner.py` 是唯一的程序入口：`build_graph()` 在模块级只编译一次，`run_once` 只传**本轮**用户消息，历史由 checkpointer 按 `thread_id` 还原——不要自己拼接历史，会重复。

## 评测

```bash
# 全量调优集，温度 0，每条 3 次
$env:MODEL_TEMPERATURE="0"
.\venv\Scripts\python.exe eval\eval_e2e.py --repeat 3

# 留出集（换数据集 + 换输出文件）
.\venv\Scripts\python.exe eval\eval_e2e.py --set eval\eval_set_holdout.jsonl --out eval\holdout_result.json --repeat 3

# 先小样本确认链路
.\venv\Scripts\python.exe eval\eval_e2e.py --limit 10
```

四个指标：

| 指标 | 算法 | 健康量级 |
| --- | --- | --- |
| 通过率 | 期望工具全部调到 且 `key_points` 全部命中 | ≥0.85 |
| 检索执行率 | policy / multi_hop 样本中路由判对的比例 | 接近 1.0 |
| 含未溯源数字率 | 答案里的业务数字（≤4 位）在证据中找不到的样本占比 | ≤0.1 |
| 平均轮次 | 每条样本 AIMessage 数量均值 | 2–3，>4 说明工具描述有问题 |

**数字溯源检测**是整个评测里最有用的一项：把答案里每个数字拿去和工具返回、检索片段比对，找不到就记疑似幻觉。售后场景里编造“15 日”“9 票”“650 分”是硬伤，说错一个数字就可能让用户白跑一趟。订单号、运单号、工单号这类长数字串是真实凭证，不计入。

评测集配置（`eval_set.jsonl` 45 条）：policy 12 / multi_hop 11 / tool 9 / boundary 8 / chitchat 5。留出集 20 条独立维护，**冻结不改**——一旦为了让它通过而调词表，它就退化成训练集了。

## 目录

```
├── RAG/rag_tools.py              规则库：加载 docx → 切块 → 向量化 → 混合检索 → search_knowledge
├── agent/
│   ├── state.py                  AgentState 定义
│   ├── lexicon.py                关键词分类（boundary / policy / other）
│   ├── nodes.py                  四个节点 + system prompt
│   ├── tools.py                  10 个工具与数据加载
│   └── graph.py                  StateGraph 装配
├── runner.py                     统一入口：run_once(question, thread_id)
├── server.py                     FastAPI + SSE 服务层
├── web/                          原生 JS 前端，无构建
├── model.py                      ChatOpenAI 工厂（超时 60s、重试 2 次）
├── orders.json                   15 笔订单（唯一数据源）
├── refunds.json                  2 笔退款单
├── disputes.json                 2 笔纠纷
├── ShopMate-闲鱼售后规则详细版.docx   规则库原文，6 章 1 万字
├── chroma_db/                    Chroma 持久化（collection: shopmate，30 块）
├── eval/
│   ├── eval_e2e.py               端到端评测
│   ├── eval_set.jsonl            45 条调优集
│   ├── eval_set_holdout.jsonl    20 条留出集（冻结）
│   ├── BADCASES.md               三个 bad case + 两个附录
│   └── eval_result*.json         各版本明细，可复现轨迹
├── old_agent/run_agent.py        create_agent 旧基线（对照用，不参与线上）
├── pyrightconfig.json            类型检查指向 venv
└── requirements.txt
```

## 检索参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| 切块 | 400 字 / 重叠 50 | `RecursiveCharacterTextSplitter` |
| 向量库 | Chroma，collection `shopmate` | 30 块，仅规则文档 |
| 稠密检索 | `text-embedding-v3`，k=5 | 1024 维，必须 `check_embedding_ctx_length=False` + `chunk_size=10`（通义单批上限 10 条） |
| 稀疏检索 | BM25 + jieba 分词，k=5 | 中文必须先分词，否则召回为 0 |
| 融合 | `EnsembleRetriever`，权重 0.5/0.5 | RRF 融合，结果不带 score |

## 技术栈

- LangChain 1.4：`@tool`、`ChatOpenAI`、`RecursiveCharacterTextSplitter`、`EnsembleRetriever`
- LangGraph 1.2：`StateGraph`、`ToolNode`、`tools_condition`、`add_messages` reducer、`InMemorySaver`
- 向量：Chroma 1.5 + 通义 `text-embedding-v3`
- 服务：FastAPI 0.162 + SSE，前端原生 JS
- 可观测：LangSmith（已接入，trace 可直接看每轮路由/检索/工具参数）

## 已知限制

- **关键词路由有上限。** 词表覆盖不到的说法（“部分退款”“折价”“瑕疵”）会漏判成 `other`，导致规则检索不触发。留出集那条失败样本就是这个原因。下一步应换成语义分类（few-shot 分类调用或 embedding 匹配问法示例）。
- **记忆只到进程级。** `InMemorySaver` 一退出就丢；换 `SqliteSaver` 只需改 `graph.py` 一行，`thread_id` 机制不动。CLI 里 `thread_id` 写死 `cli-1`，线上必须按用户/会话分配，否则不同用户会共享记忆。
- **数据是 mock 的。** 15 笔订单、2 笔退款、2 笔纠纷，规模只能支撑评测，不代表真实分布。
- **评测集是自建的。** 45 条调优集在优化过程中被反复看过，97.78% 属于训练集成绩；对外应引用留出集的 95%。

## 注意

`AgentExecutor`、`create_react_agent` 均已弃用，网上大量教程仍在用，照着写会踩坑。本项目用 `StateGraph` 显式编排，用 `create_agent` 的旧写法保留在 `old_agent/` 作对照基线。
