"""ShopMate 售后服务台 —— FastAPI + SSE 服务层。

把 agent/graph.py 里的 LangGraph 图包装成浏览器可用的流式接口：
    POST /api/chat   -> text/event-stream，逐事件推送 route / retrieve / tool / delta
    GET  /api/orders -> 订单列表，供前端侧边栏选择
    GET  /api/health -> 服务与 Agent 就绪状态

启动：
    ./venv/Scripts/python.exe server.py
然后打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

ORDERS: list[dict[str, Any]] = json.loads((BASE_DIR / "orders.json").read_text("utf-8"))
ORDER_MAP = {o["订单号"]: o for o in ORDERS}

# Agent 图是懒加载单例：导入 RAG 链要加载 docx、embedding 和 BM25，耗时数秒，
# 不能让第一个 HTTP 请求去等它。
_graph = None
_graph_lock = threading.Lock()


def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from agent.graph import build_graph

                _graph = build_graph()
    return _graph


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_safe_warmup, daemon=True).start()
    yield


def _safe_warmup() -> None:
    try:
        get_graph()
    except Exception as exc:  # noqa: BLE001 - 预热失败不应阻断服务启动
        print(f"[warmup] Agent 预热失败：{exc}")


app = FastAPI(title="ShopMate 售后服务台", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(default="default", max_length=64)


def _sse(event: str, data: dict[str, Any]) -> str:
    body = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def _text_of(message: Any) -> str:
    """兼容 str / content blocks 两种 content 形态。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


def _clip(text: str, limit: int = 700) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def _stream_chat(message: str, thread_id: str) -> Iterator[str]:
    yield _sse("meta", {"thread_id": thread_id, "ts": time.time()})

    try:
        graph = get_graph()
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": f"Agent 初始化失败：{exc}"})
        yield _sse("done", {"answer": "", "route": None, "escalated": False})
        return

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    inputs = {"messages": [{"role": "user", "content": message}]}

    chunks: list[str] = []
    last_answer = ""
    route: str | None = None
    escalated = False

    try:
        # updates 提供节点级过程；messages 提供模型 token 级输出，两者合流。
        for mode, payload in graph.stream(
            inputs, config=config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, meta = payload
                if (meta or {}).get("langgraph_node") != "agent":
                    continue
                piece = _text_of(chunk)
                if piece:
                    chunks.append(piece)
                    yield _sse("delta", {"text": piece})
                continue

            for node, update in (payload or {}).items():
                if node == "route":
                    route = update.get("route")
                    yield _sse("route", {"route": route, "order_id": update.get("order_id")})
                elif node == "retrieve":
                    ctx = update.get("retrieved") or ""
                    yield _sse(
                        "retrieval",
                        {
                            "chars": len(ctx),
                            "preview": _clip(ctx[:200].replace("\n", " "), 180),
                        },
                    )
                elif node == "agent":
                    answer_msg = (update.get("messages") or [None])[-1]
                    last_answer = _text_of(answer_msg) or last_answer
                    for call in getattr(answer_msg, "tool_calls", None) or []:
                        yield _sse(
                            "tool_call",
                            {
                                "name": call.get("name") or "",
                                "args": call.get("args") or {},
                                "id": call.get("id") or "",
                            },
                        )
                elif node == "tools":
                    for tool_msg in update.get("messages") or []:
                        yield _sse(
                            "tool_result",
                            {
                                "name": getattr(tool_msg, "name", "") or "",
                                "content": _clip(_text_of(tool_msg)),
                            },
                        )
                elif node == "handoff":
                    escalated = True
                    text = _text_of((update.get("messages") or [None])[-1])
                    if text:
                        chunks.append(text)
                        yield _sse("delta", {"text": text})
                    yield _sse("escalate", {"text": text})
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": f"执行出错：{exc}"})

    answer = "".join(chunks).strip() or last_answer
    yield _sse("done", {"answer": answer, "route": route, "escalated": escalated})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "agent_ready": _graph is not None}


@app.get("/api/orders")
def list_orders(q: str = Query(default="", max_length=64)) -> dict[str, Any]:
    keyword = q.strip().upper()
    rows: list[dict[str, Any]] = []
    for order in ORDERS:
        items = order.get("items") or []
        names = "、".join(str(it[1]) for it in items if len(it) > 1)
        if keyword and not any(
            keyword in field.upper()
            for field in (order["订单号"], order["订单状态"], names)
        ):
            continue
        rows.append(
            {
                "order_id": order["订单号"],
                "status": order["订单状态"],
                "amount": order["订单金额"],
                "items": names,
                "carrier": order.get("carrier") or "—",
                "tracking_no": order.get("tracking_no") or "—",
            }
        )
    return {"total": len(rows), "items": rows}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    thread_id = req.thread_id.strip() or "default"
    return StreamingResponse(
        _stream_chat(req.message.strip(), thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reset")
def reset(thread_id: str = Query(default="", max_length=64)) -> dict[str, Any]:
    """清空某个会话的记忆（InMemorySaver 按 thread_id 存历史）。"""
    graph = get_graph()
    try:
        saver = graph.checkpointer
        if saver is not None and thread_id:
            saver.delete_thread(thread_id)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 内存 checkpointer 版本差异时忽略
        pass
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import os

    import uvicorn

    # 默认只监听本机。要让同一局域网的其他人访问，用 SHOPMATE_HOST=0.0.0.0 启动。
    host = os.getenv("SHOPMATE_HOST", "127.0.0.1")
    port = int(os.getenv("SHOPMATE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
