import json
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool

from RAG.rag_tools import search_knowledge


def transfer_to_human_impl(reason: str, user_message: str) -> str:
    ticket = "HR20261030001"
    return f"已为您转接人工客服，工单号 {ticket}，预计 5 分钟内接入。触发原因：{reason}"


ORDERS = {o["订单号"]: o for o in json.loads(
    (Path(__file__).parent.parent / "orders.json").read_text("utf-8"))}

_BASE = Path(__file__).resolve().parent.parent


def _load_map(filename: str) -> dict:
    """加载按订单号索引的 JSON；文件不存在时返回空字典，保证工具不崩。"""
    p = _BASE / filename
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text("utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict):
        return raw
    return {(x.get("订单号") or x.get("order_id")): x for x in raw if isinstance(x, dict)}


REFUNDS = _load_map("refunds.json")
DISPUTES = _load_map("disputes.json")
GUARANTEES = _load_map("guarantees.json")

WINDOW_DAYS = 15
_NOT_SHIPPED = {"待付款", "待发货"}
FREIGHT_BACK = 12.0
INSPECTION_FEE = 29.0
REFUND_TYPES = {
    "item_issue": "假货/错发/破损/描述不符",
    "buyer_reason": "不喜欢/拍错/尺寸不合",
    "inspection_mismatch": "验货宝鉴定不符",
    "not_shipped": "未发货",
    "virtual": "虚拟商品",
    "custom": "定制商品",
}


def _err(code: str, hint: str) -> str:
    return json.dumps({"ok": False, "error": code, "hint": hint}, ensure_ascii=False)


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


@tool
def query_orders(order_id: str) -> str:
    """查订单事实：状态、实付金额、商品明细、签收时间、收货地址。
    用户问「这单什么情况」「多少钱」「买的什么」时调用。
    问物流轨迹用 query_logistics，问售后时效用 get_after_sale_window。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号形如 ORD 加 12 位数字，请向用户核对，不要编造。")

    items = []
    for it in (o.get("items") or []):
        if isinstance(it, (list, tuple)):
            items.append({"sku": it[0] if len(it) > 0 else None,
                          "title": it[1] if len(it) > 1 else None,
                          "qty": it[2] if len(it) > 2 else None,
                          "price": it[3] if len(it) > 3 else None})
        elif isinstance(it, dict):
            items.append(it)

    return json.dumps({"ok": True, "order_id": oid,
                       "status": o.get("订单状态"),
                       "paid_amount": float(o.get("订单金额") or 0),
                       "items": items,
                       "seller_type": o.get("卖家类型") or "个人",
                       "signed_at": o.get("签收时间") or o.get("signed_at"),
                       "address": o.get("收货地址")}, ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """查物流：承运商、运单号、最新更新时间。用户问「到哪了」「发货了吗」「什么时候到」时调用。
    问金额、商品、售后时效请勿调用。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号形如 ORD 加 12 位数字，请向用户核对，不要编造。")

    carrier = (o.get("carrier") or "").strip()
    tracking = (o.get("tracking_no") or "").strip()
    if not carrier and not tracking:
        return json.dumps({"ok": True, "order_id": oid, "has_logistics": False,
                           "order_status": o.get("订单状态"),
                           "hint": "该订单暂无物流信息，可能尚未发货。"},
                          ensure_ascii=False)

    return json.dumps({"ok": True, "order_id": oid, "has_logistics": True,
                       "carrier": carrier, "tracking_no": tracking,
                       "latest_time": o.get("logistics_time"),
                       "order_status": o.get("订单状态")}, ensure_ascii=False)


@tool
def get_after_sale_window(order_id: str) -> str:
    """查售后时效：签收时间、已过天数、剩余天数、是否还能申请售后。
    用户问「还能退吗」「过售后期了吗」「还剩几天」时调用。
    天数以本工具返回值为准，禁止自己按日期推算。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号形如 ORD 加 12 位数字，请向用户核对。")

    status = o.get("订单状态") or ""
    if status in _NOT_SHIPPED:
        return json.dumps({"ok": True, "order_id": oid, "can_apply": None,
                           "order_status": status, "window_days": WINDOW_DAYS,
                           "reason": "订单尚未发货，售后期自签收后开始计算。",
                           "next_step": "可引导用户直接申请取消订单/未发货退款。"},
                          ensure_ascii=False)

    signed_at = _parse_dt(o.get("签收时间") or o.get("signed_at"))
    if signed_at is None:
        return json.dumps({"ok": True, "order_id": oid, "can_apply": None,
                           "order_status": status, "window_days": WINDOW_DAYS,
                           "reason": "系统中缺少该订单的签收时间，无法判定售后期。",
                           "next_step": "转人工核实，不要猜测是否过期。"},
                          ensure_ascii=False)

    signed: datetime = signed_at
    days = max((datetime.now() - signed).days, 0)
    remaining = WINDOW_DAYS - days
    return json.dumps({"ok": True, "order_id": oid,
                       "signed_at": signed.strftime("%Y-%m-%d %H:%M:%S"),
                       "days_since_signed": days,
                       "window_days": WINDOW_DAYS,
                       "remaining_days": remaining,
                       "deadline": (signed + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d"),
                       "can_apply": remaining >= 0,
                       "reason": (f"签收第 {days} 天，仍在 {WINDOW_DAYS} 日售后期内。"
                                  if remaining >= 0 else
                                  f"已超出 {WINDOW_DAYS} 日售后期 {-remaining} 天。")},
                      ensure_ascii=False)


_ISSUE_HINTS = (
    ("not_shipped", ("没发货", "未发货", "不发货", "还没发", "一直不", "催发货")),
    ("item_issue", ("假", "破损", "坏了", "坏", "质量", "错发", "发错", "漏发",
                    "少发", "不符", "描述", "有问题", "瑕疵", "污", "空包")),
    ("buyer_reason", ("不喜欢", "不需要", "拍错", "买错", "尺寸", "不合", "后悔", "多买")),
)


def _guess_refund_type(oid: str, issue_desc: str) -> tuple[str, str]:
    """用户还没申请退款、拿不到 refund_type 时，按「已登记退款单 > 用户描述 > 订单状态」推断。
    返回 (refund_type, 推断依据)。"""
    rec = REFUNDS.get(oid) or {}
    t = str(rec.get("refund_type") or "").strip().lower()
    if t in REFUND_TYPES:
        return t, "该订单已有退款单，取登记的类型"

    desc = (issue_desc or "").strip()
    for guess, kws in _ISSUE_HINTS:
        if any(k in desc for k in kws):
            return guess, f"按用户描述「{desc}」推断"

    status = str((ORDERS.get(oid) or {}).get("订单状态") or "")
    if status in _NOT_SHIPPED:
        return "not_shipped", f"订单状态「{status}」，按未发货试算"
    return "item_issue", f"订单状态「{status}」，未说明原因，按商品问题试算"


@tool
def calc_refund_amount(order_id: str, refund_type: str = "", issue_desc: str = "") -> str:
    """按售后类型试算可退金额，逐项列出扣减/加项、运费承担方、资金状态。
    用户问「能退多少」「运费谁出」「鉴定费退不退」时调用。
    refund_type 优先取 query_refund_status 的返回；该订单还没申请退款时，
    把用户原话里描述的原因填进 issue_desc（如「商品有问题」「不喜欢了」「一直不发货」），
    工具会自行推断类型并在 basis 里标注依据。
    不要因为"查不到退款单"就拒绝试算或转人工。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号不存在，请先确认订单。")

    t = (refund_type or "").strip().lower()
    if t in REFUND_TYPES:
        basis, estimated = "调用方指定", False
    else:
        t, basis = _guess_refund_type(oid, issue_desc or refund_type or "")
        estimated = True

    if t in ("virtual", "custom"):
        return _err("NOT_REFUNDABLE", f"{REFUND_TYPES[t]}不支持无理由退款，依据见规则库。")

    status = o.get("订单状态") or ""
    paid = round(float(o.get("订单金额") or 0), 2)
    items, payer = [], "卖家"

    if t == "not_shipped":
        if status not in _NOT_SHIPPED:
            return _err("STATUS_MISMATCH",
                        f"该订单状态为「{status}」，不是未发货，请先核对 refund_type。")
        final = paid
        note = "未发货，全额退款，不扣任何费用。"
    elif t == "buyer_reason":
        items.append({"name": "回寄运费", "amount": -FREIGHT_BACK,
                      "note": "买家主观原因，需卖家同意后扣除"})
        payer, final = "买家", round(paid - FREIGHT_BACK, 2)
        note = "买家主观原因，扣除回寄运费。"
    elif t == "inspection_mismatch":
        items.append({"name": "验货宝鉴定费", "amount": INSPECTION_FEE,
                      "note": "鉴定不符，鉴定费由平台一并退回（加项，不是扣款）"})
        final = round(paid + INSPECTION_FEE, 2)
        note = "鉴定不符，货款全退并加退鉴定费。"
    else:  # item_issue
        final = paid
        note = "商品问题，货款全额退还，运费由卖家承担。"

    if estimated:
        note += f"（该订单尚未发起退款，本结果为按「{REFUND_TYPES[t]}」预估，实际以申请后审核为准）"

    return json.dumps({"ok": True, "order_id": oid,
                       "refund_type": t,
                       "basis": basis,
                       "estimated": estimated,
                       "paid_amount": paid,
                       "deduction_items": items,
                       "final_amount": final,
                       "freight_payer": payer,
                       "fund_status": "担保中，可直接退回" if status != "已完成" else "已结算，需人工处理",
                       "notes": note}, ensure_ascii=False)


@tool
def query_refund_status(order_id: str) -> str:
    """查退款进度：是否已申请、退款类型、申请金额、当前环节、截止时间。
    用户问「退款到哪了」「申请通过了吗」「卖家不处理怎么办」时调用。
    只咨询退款规则时请勿调用（应查知识库）。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号形如 ORD 加 12 位数字，请向用户核对。")

    r = REFUNDS.get(oid)
    if not r:
        return json.dumps({"ok": True, "applied": False, "order_id": oid,
                           "order_status": o.get("订单状态"),
                           "hint": "该订单尚未申请退款。可引导用户在订单页「申请售后」发起。"},
                          ensure_ascii=False)

    d = r.get("dispute") or {}
    return json.dumps({"ok": True, "applied": True, "order_id": oid,
                       "refund_status": r.get("refund_status") or r.get("退款状态"),
                       "refund_type": r.get("refund_type") or r.get("退款类型"),
                       "refund_amount": r.get("refund_amount") or r.get("退款金额"),
                       "applied_at": r.get("applied_at") or r.get("申请时间"),
                       "current_step": r.get("current_step") or r.get("当前环节"),
                       "deadline": d.get("deadline") or r.get("deadline")},
                      ensure_ascii=False)


_GUARANTEE_BY_STATUS = {
    "待付款": ("未支付", "买家尚未付款，无担保资金。"),
    "待发货": ("担保中", "货款由平台代管，卖家发货后继续冻结。"),
    "已发货": ("担保中", "货款由平台代管，买家确认收货后放款给卖家。"),
    "已签收": ("担保中", "货款仍在平台代管，售后期结束且无纠纷后放款。"),
    "退款中": ("冻结中", "订单处于退款流程，资金暂停结算。"),
    "已完成": ("已放款", "订单已完成，货款已结算给卖家。"),
    "已退款": ("已退回", "款项已退回买家。"),
    "已关闭": ("已退回", "交易关闭，款项已退回买家。"),
    "已取消": ("已退回", "订单已取消，货款由平台原路退回买家。"),
}


@tool
def query_guarantee(order_id: str) -> str:
    """查担保资金：钱现在在谁手上、金额多少、什么条件下放款给卖家。
    用户问「钱在谁手上」「卖家收到钱了吗」「担保金退了吗」时调用。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号形如 ORD 加 12 位数字，请向用户核对。")

    g = GUARANTEES.get(oid) or {}
    status = g.get("status") or g.get("担保状态")
    reason = g.get("release_condition") or g.get("放款条件")
    inferred = False
    if not status:
        status, reason = _GUARANTEE_BY_STATUS.get(
            o.get("订单状态") or "", ("未知", "该状态无对应规则，需人工核实。"))
        inferred = True

    return json.dumps({"ok": True, "order_id": oid,
                       "held": status in ("担保中", "冻结中"),
                       "status": status,
                       "amount": round(float(g.get("amount") or o.get("订单金额") or 0), 2),
                       "release_condition": reason,
                       "inferred": inferred}, ensure_ascii=False)


@tool
def get_dispute_status(order_id: str) -> str:
    """查纠纷/小法庭进度：当前阶段、举证截止、双方票数、下一步动作。
    用户问「卖家不处理怎么办」「小法庭到哪了」「客服介入结果」时调用。"""
    oid = (order_id or "").strip().upper()
    o = ORDERS.get(oid)
    if not o:
        return _err("ORDER_NOT_FOUND", "订单号形如 ORD 加 12 位数字，请向用户核对。")

    d = DISPUTES.get(oid)
    if not d:
        return json.dumps({"ok": True, "order_id": oid, "stage": None,
                           "order_status": o.get("订单状态"),
                           "hint": "该订单当前没有进行中的纠纷。若卖家拒绝退款，可在退款页申请客服介入。"},
                          ensure_ascii=False)

    return json.dumps({"ok": True, "order_id": oid,
                       "stage": d.get("stage") or d.get("阶段"),
                       "evidence_deadline": d.get("evidence_deadline") or d.get("举证截止"),
                       "votes_buyer": d.get("votes_buyer"),
                       "votes_seller": d.get("votes_seller"),
                       "next_step": d.get("next_step") or d.get("下一步"),
                       "result": d.get("result") or d.get("裁决结果")},
                      ensure_ascii=False)


@tool
def get_current_time():
    """返回当前的日期和时间。当用户提到「今天几号」「现在几点」时调用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def transfer_to_human(user_message: str) -> str:
    """转人工。仅当用户明确要求人工时调用。"""
    return transfer_to_human_impl(reason="model_requested", user_message=user_message)


TOOLS = [
    get_current_time,
    query_orders,
    query_logistics,
    transfer_to_human,
    search_knowledge,
    get_after_sale_window,
    calc_refund_amount,
    query_refund_status,
    query_guarantee,
    get_dispute_status,
]
