"""端到端评测：通过率 / 检索执行率 / 未溯源数字率 / 平均轮次。

用法：
    .\\venv\\Scripts\\python.exe eval\\eval_e2e.py --limit 10
    .\\venv\\Scripts\\python.exe eval\\eval_e2e.py
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import ToolMessage

from RAG.rag_tools import retriever
from runner import run_once

SET_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "eval_result.json"
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def load(limit: int = 0, path: Path = SET_PATH) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    if not limit or limit >= len(rows):
        return rows
    # 分层抽样：每类轮流取，避免 --limit 10 全落在 policy 上
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["category"], []).append(r)
    out, i = [], 0
    while len(out) < limit:
        added = False
        for items in buckets.values():
            if i < len(items) and len(out) < limit:
                out.append(items[i])
                added = True
        if not added:
            break
        i += 1
    return out


def evidence(rec: dict, question: str) -> str:
    """证据 = 工具返回 + 检索片段，用于判断答案里的数字有没有出处。"""
    parts = [str(m.content) for m in rec["messages"] if isinstance(m, ToolMessage)]
    try:
        parts += [d.page_content for d in retriever.invoke(question)]
    except Exception:
        pass
    return "\n".join(parts)


def judge(sample: dict, rec: dict) -> dict:
    answer = str(rec["answer"])
    cat = sample["category"]
    exp_tools = sample.get("expect_tools") or []
    called = rec["tool_calls"]

    missing = [t for t in exp_tools if t not in called]
    extra = [t for t in called if t not in exp_tools]

    # 边界类不看工具：handoff 由节点直接执行，不走工具调用
    if cat == "boundary":
        tool_ok = bool(rec["escalated"])
    elif cat == "chitchat":
        tool_ok = not called
    else:
        tool_ok = not missing

    miss_points = [p for p in sample["key_points"] if p not in answer]
    ev = evidence(rec, sample["question"])
    # 只查"业务数字"（金额/天数/票数/比例）。订单号、工单号、运单号这类长数字串是真实凭证，不算幻觉
    numbers = [n for n in NUM_RE.findall(answer) if len(n.split(".")[0]) <= 4]
    untraced = [n for n in numbers if n not in ev]

    return {
        "id": sample["id"],
        "category": cat,
        "passed": tool_ok and not miss_points,
        "tool_ok": tool_ok,
        "missing_tools": missing,
        "extra_tools": extra,
        "miss_points": miss_points,
        "route_ok": rec["route"] == sample["expect_route"],
        "route": rec["route"],
        "numbers": numbers,
        "untraced_numbers": untraced,
        "turns": rec["turns"],
        "answer": answer[:300],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（按类别分层抽样）")
    ap.add_argument("--repeat", type=int, default=1, help="每条样本重复跑几次，多数通过才算通过")
    ap.add_argument("--set", default=str(SET_PATH), help="评测集路径，默认 eval_set.jsonl")
    ap.add_argument("--out", default=str(OUT_PATH), help="结果输出路径，默认 eval_result.json")
    args = ap.parse_args()

    need = args.repeat // 2 + 1          # repeat=1→1，2→2，3→2
    samples = load(args.limit, Path(args.set))
    results = []
    for s in samples:
        runs = []
        for _ in range(args.repeat):
            rec = run_once(s["question"], thread_id=f'{s["id"]}-{uuid.uuid4().hex[:6]}')
            runs.append((rec, judge(s, rec)))

        pass_runs = sum(1 for _, j in runs if j["passed"])
        rec, r = runs[0]
        r["pass_runs"] = pass_runs
        r["repeat"] = args.repeat
        r["passed"] = pass_runs >= need
        results.append(r)

        mark = "PASS" if r["passed"] else "FAIL"
        print(f'{mark} {r["id"]} {r["category"]:<9} {pass_runs}/{args.repeat} '
              f'route={str(r["route"]):<8} tools={rec["tool_calls"]}')
        if not r["passed"]:
            print(f'     missing={r["missing_tools"]} points={r["miss_points"]} '
                  f'untraced={r["untraced_numbers"]}')

    total = len(results)
    passed = sum(r["passed"] for r in results)
    retrieval_pool = [r for r in results if r["category"] in ("policy", "multi_hop")]
    retrieval_rate = (
        sum(r["route_ok"] for r in retrieval_pool) / len(retrieval_pool)
        if retrieval_pool else 0.0
    )
    with_num = [r for r in results if r["numbers"]]
    halluc_rate = (
        sum(1 for r in with_num if r["untraced_numbers"]) / len(with_num)
        if with_num else 0.0
    )
    avg_turns = sum(r["turns"] for r in results) / total if total else 0.0

    summary = {
        "samples": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "retrieval_rate": round(retrieval_rate, 4),
        "untraced_rate": round(halluc_rate, 4),
        "avg_turns": round(avg_turns, 2),
    }

    print("\n" + "=" * 46)
    print(f'样本数        {summary["samples"]}')
    print(f'通过率        {summary["pass_rate"]:.2%}  ({passed}/{total})')
    print(f'检索执行率    {summary["retrieval_rate"]:.2%}')
    print(f'含未溯源数字  {summary["untraced_rate"]:.2%}')
    print(f'平均轮次      {summary["avg_turns"]}')
    print("=" * 46)

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2),
        "utf-8",
    )
    print(f"明细已写入 {out_path}")

if __name__ == "__main__":
    main()
