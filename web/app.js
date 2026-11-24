/* ShopMate 售后服务台 · 前端逻辑
   流式对话（SSE）+ 会话管理 + 订单侧栏 + 执行轨迹展示 */

const $ = (sel, root = document) => root.querySelector(sel);

const STORAGE_KEY = "shopmate.chats.v2";
const ORDER_RE = /ORD\d{12}/i;

const ROUTE_LABEL = {
  policy: "规则咨询",
  boundary: "边界拦截",
  other: "常规问答",
};

const TOOL_LABEL = {
  query_orders: "查询订单",
  query_logistics: "查询物流",
  search_knowledge: "检索规则库",
  get_current_time: "获取当前时间",
  transfer_to_human: "转人工",
};

const els = {
  messages: $("#messages"),
  welcome: $("#welcome"),
  input: $("#input"),
  send: $("#send"),
  newChat: $("#newChat"),
  clearChat: $("#clearChat"),
  routeBadge: $("#routeBadge"),
  orderChip: $("#orderChip"),
  chatPanel: $("#chatPanel"),
  orderPanel: $("#orderPanel"),
  orderList: $("#orderList"),
  orderSearch: $("#orderSearch"),
  healthDot: $("#healthDot"),
  healthText: $("#healthText"),
};

const state = {
  threadId: null,
  chats: [],
  orders: [],
  streaming: false,
  current: null, // 当前 assistant 节点
};

/* ------------------------------------------------------------ 存储 */

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const data = raw ? JSON.parse(raw) : null;
    if (data && Array.isArray(data.chats)) state.chats = data.chats;
  } catch {
    state.chats = [];
  }
}

function saveStore() {
  try {
    const slim = state.chats.slice(0, 20).map((c) => ({
      ...c,
      messages: c.messages.slice(-40),
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ chats: slim }));
  } catch {
    /* 存储满了就静默忽略 */
  }
}

function currentChat() {
  return state.chats.find((c) => c.id === state.threadId) || null;
}

function newThread() {
  const id = "t-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  state.chats.unshift({ id, title: "新会话", updatedAt: Date.now(), messages: [] });
  state.threadId = id;
  renderChatList();
  renderThread();
  saveStore();
}

/* ------------------------------------------------------------ 渲染：会话列表 */

function renderChatList() {
  els.chatPanel.innerHTML = "";
  if (!state.chats.length) {
    els.chatPanel.innerHTML = '<div class="empty-hint">还没有会话</div>';
    return;
  }
  for (const chat of state.chats) {
    const btn = document.createElement("button");
    btn.className = "chat-item" + (chat.id === state.threadId ? " active" : "");
    btn.innerHTML = `
      <span class="chat-title"></span>
      <span class="chat-del" title="删除会话">×</span>`;
    btn.querySelector(".chat-title").textContent = chat.title || "新会话";
    btn.addEventListener("click", (e) => {
      if (e.target.classList.contains("chat-del")) {
        e.stopPropagation();
        removeChat(chat.id);
        return;
      }
      switchThread(chat.id);
    });
    els.chatPanel.appendChild(btn);
  }
}

function removeChat(id) {
  state.chats = state.chats.filter((c) => c.id !== id);
  if (state.threadId === id) {
    if (state.chats.length) switchThread(state.chats[0].id);
    else newThread();
  } else {
    renderChatList();
  }
  saveStore();
}

function switchThread(id) {
  if (state.streaming) return;
  state.threadId = id;
  renderChatList();
  renderThread();
  saveStore();
}

/* ------------------------------------------------------------ 渲染：消息 */

function renderThread() {
  els.messages.querySelectorAll(".msg").forEach((n) => n.remove());
  const chat = currentChat();
  const msgs = chat ? chat.messages : [];

  if (!msgs.length) {
    els.welcome.classList.remove("hidden");
    updateRouteBadge("");
    return;
  }

  els.welcome.classList.add("hidden");
  let lastRoute = "";
  for (const m of msgs) {
    if (m.role === "user") appendUserMessage(m.text, false);
    else appendAssistantMessage(m);
    if (m.route) lastRoute = m.route;
  }
  updateRouteBadge(lastRoute);
  scrollToBottom(true);
}

function appendUserMessage(text, animate = true) {
  const el = document.createElement("div");
  el.className = "msg user";
  if (!animate) el.style.animation = "none";
  el.innerHTML = `
    <div class="bubble-avatar">我</div>
    <div class="msg-body">
      <div class="bubble"></div>
      <div class="msg-meta"></div>
    </div>`;
  el.querySelector(".bubble").textContent = text;
  el.querySelector(".msg-meta").textContent = timeLabel();
  els.messages.appendChild(el);
  scrollToBottom();
  return el;
}

function appendAssistantMessage(msg) {
  const node = createAssistantNode(false);
  node.setText(msg.text || "");
  node.meta.textContent = msg.escalated ? "已转人工 · " + timeLabel() : timeLabel();
  for (const step of msg.steps || []) node.pushStep(step);
  node.finish();
  return node;
}

function createAssistantNode(animate = true) {
  const el = document.createElement("div");
  el.className = "msg assistant";
  if (!animate) el.style.animation = "none";
  el.innerHTML = `
    <div class="bubble-avatar">S</div>
    <div class="msg-body">
      <div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>
      <div class="trace hidden">
        <button class="trace-head" type="button">
          <span class="trace-icon"><svg viewBox="0 0 24 24"><path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z"/></svg></span>
          <span class="trace-sum">执行轨迹</span>
          <span class="trace-count">0 步</span>
          <span class="caret">▾</span>
        </button>
        <div class="trace-steps"></div>
      </div>
      <div class="msg-meta"></div>
    </div>`;

  const bubble = el.querySelector(".bubble");
  const trace = el.querySelector(".trace");
  const stepsBox = el.querySelector(".trace-steps");
  const count = el.querySelector(".trace-count");
  const meta = el.querySelector(".msg-meta");
  let text = "";
  let dirty = false;
  let steps = [];

  el.querySelector(".trace-head").addEventListener("click", () => {
    trace.classList.toggle("open");
  });

  const paint = () => {
    if (!dirty) return;
    dirty = false;
    bubble.innerHTML = renderMarkdown(text) + '<span class="cursor"></span>';
    scrollToBottom();
  };

  return {
    el,
    meta,
    trace,
    stepsBox,
    getText: () => text,
    getSteps: () => steps,
    setText(value) {
      text = value || "";
      if (!text) return;
      dirty = true;
      requestAnimationFrame(paint);
    },
    appendText(piece) {
      if (!piece) return;
      text += piece;
      dirty = true;
      requestAnimationFrame(paint);
    },
    finish() {
      dirty = false;
      bubble.innerHTML = renderMarkdown(text || "（无回复）");
      scrollToBottom();
    },
    pushStep(step) {
      steps.push(step);
      trace.classList.remove("hidden");
      count.textContent = `${steps.length} 步`;
      const node = document.createElement("div");
      node.className = "step";
      node.dataset.kind = step.kind;
      if (step.tool) node.dataset.tool = step.tool;
      const args = step.args
        ? Object.entries(step.args)
            .map(([k, v]) => `<span class="kv">${escapeHtml(k)}: ${escapeHtml(String(v))}</span>`)
            .join("")
        : "";
      node.innerHTML = `
        <div class="step-name">${escapeHtml(step.name)}</div>
        ${step.detail ? `<div class="step-detail">${escapeHtml(step.detail)}</div>` : ""}
        ${args ? `<div class="step-detail">${args}</div>` : ""}
        <div class="step-detail result-slot"></div>`;
      const slot = node.querySelector(".result-slot");
      if (step.result) {
        const box = document.createElement("div");
        box.className = "result";
        box.textContent = step.result;
        slot.appendChild(box);
      }
      stepsBox.appendChild(node);
      scrollToBottom();
    },
    attachResult(name, content) {
      const step =
        [...steps].reverse().find((s) => s.tool === name && !s.result) ||
        [...steps].reverse().find((s) => !s.result);
      if (step) step.result = content;

      const nodes = [...stepsBox.querySelectorAll(".step")];
      const node =
        [...nodes].reverse().find((n) => n.dataset.tool === name && !n.querySelector(".result")) ||
        [...nodes].reverse().find((n) => !n.querySelector(".result"));
      if (!node) return;

      const box = document.createElement("div");
      box.className = "result";
      box.textContent = content;
      node.querySelector(".result-slot").appendChild(box);
      scrollToBottom();
    },
  };
}

function updateRouteBadge(route) {
  els.routeBadge.dataset.route = route || "";
  els.routeBadge.textContent = route ? ROUTE_LABEL[route] || route : "待命";
}

function timeLabel() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom(force = false) {
  const el = els.messages;
  const near = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
  if (force || near) el.scrollTop = el.scrollHeight;
}

/* ------------------------------------------------------------ 极简 Markdown */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(src) {
  let text = escapeHtml(src);
  const blocks = [];
  text = text.replace(/```[\s\S]*?```/g, (m) => {
    blocks.push(`<pre><code>${m.slice(3, -3).trim()}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });

  text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");

  const lines = text.split("\n");
  const out = [];
  let list = null;

  const closeList = () => {
    if (list) {
      out.push(`<ul>${list.join("")}</ul>`);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*·]\s+(.*)$/) || line.match(/^\s*\d+[.、)]\s+(.*)$/);
    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (bullet) {
      if (!list) list = [];
      list.push(`<li>${bullet[1]}</li>`);
      continue;
    }
    closeList();
    if (heading) {
      out.push(`<h3>${heading[1]}</h3>`);
    } else {
      // 普通行不包 <p>：气泡本身是 pre-wrap，包段会带来双重行距
      out.push(line);
    }
  }
  closeList();

  return out
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

/* ------------------------------------------------------------ SSE 对话 */

async function sendMessage(rawText) {
  const text = (rawText ?? els.input.value).trim();
  if (!text || state.streaming) return;
  if (!state.threadId) newThread();

  els.welcome.classList.add("hidden");
  els.input.value = "";
  autoResize();
  updateOrderChip();

  appendUserMessage(text);
  const chat = currentChat();
  chat.messages.push({ role: "user", text, ts: Date.now() });
  if (chat.title === "新会话") {
    chat.title = text.length > 18 ? text.slice(0, 18) + "…" : text;
  }
  chat.updatedAt = Date.now();
  renderChatList();
  saveStore();

  const node = createAssistantNode();
  els.messages.appendChild(node.el);
  scrollToBottom(true);
  state.current = node;
  state.streaming = true;
  setBusy(true);

  let errorText = "";
  try {
    await streamChat(text, node);
  } catch (err) {
    errorText = err?.message || "网络异常，请重试";
    node.pushStep({ kind: "escalate", name: "请求失败", detail: errorText });
    node.setText(`连接服务失败：${errorText}\n\n请确认后端已启动：python server.py`);
  }

  node.finish();
  const answer = node.getText();
  node.meta.textContent = timeLabel();

  chat.messages.push({
    role: "assistant",
    text: answer || (errorText ? `请求失败：${errorText}` : ""),
    steps: node.getSteps(),
    route: state.lastRoute || "",
    escalated: !!state.lastEscalated,
    ts: Date.now(),
  });
  chat.updatedAt = Date.now();
  saveStore();

  state.streaming = false;
  state.current = null;
  setBusy(false);
  els.input.focus();
}

async function streamChat(text, node) {
  state.lastRoute = "";
  state.lastEscalated = false;

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, thread_id: state.threadId }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      handleEvent(parseSse(raw), node);
    }
  }
}

function parseSse(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  let data = {};
  try {
    data = JSON.parse(dataLines.join("\n") || "{}");
  } catch {
    data = {};
  }
  return { event, data };
}

function handleEvent({ event, data }, node) {
  switch (event) {
    case "route": {
      state.lastRoute = data.route || "";
      updateRouteBadge(data.route);
      node.pushStep({
        kind: "route",
        name: `路由判定 · ${ROUTE_LABEL[data.route] || data.route || "未知"}`,
        detail: data.order_id ? `识别到订单号 ${data.order_id}` : "未识别到订单号",
        args: null,
      });
      break;
    }
    case "retrieval": {
      node.pushStep({
        kind: "retrieve",
        name: "检索规则知识库",
        detail: `纳入 ${data.chars || 0} 字条款依据 · ${data.preview || ""}`,
      });
      break;
    }
    case "tool_call": {
      node.pushStep({
        kind: "tool",
        tool: data.name || "",
        name: `调用工具 · ${TOOL_LABEL[data.name] || data.name}`,
        args: data.args || {},
      });
      break;
    }
    case "tool_result": {
      node.attachResult(data.name, data.content || "");
      break;
    }
    case "escalate": {
      state.lastEscalated = true;
      node.pushStep({
        kind: "escalate",
        name: "边界拦截 · 已转人工",
        detail: "未经过模型决策，由 handoff 节点直接转接",
      });
      break;
    }
    case "delta": {
      node.appendText(data.text || "");
      break;
    }
    case "error": {
      node.pushStep({ kind: "escalate", name: "服务返回错误", detail: data.message || "" });
      break;
    }
    case "done": {
      state.lastRoute = data.route || state.lastRoute;
      state.lastEscalated = !!data.escalated || state.lastEscalated;
      if (!node.getText() && data.answer) node.setText(data.answer);
      break;
    }
    default:
      break;
  }
}

/* ------------------------------------------------------------ 输入区 */

function setBusy(busy) {
  els.send.disabled = busy;
  els.input.disabled = busy;
}

function autoResize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 132) + "px";
}

function updateOrderChip() {
  const hit = els.input.value.match(ORDER_RE);
  if (hit) {
    els.orderChip.innerHTML = `已识别订单 <b>${escapeHtml(hit[0].toUpperCase())}</b>`;
    els.orderChip.classList.remove("hidden");
  } else {
    els.orderChip.classList.add("hidden");
  }
}

/* ------------------------------------------------------------ 订单面板 */

async function loadOrders() {
  try {
    const res = await fetch("/api/orders");
    const data = await res.json();
    state.orders = data.items || [];
    renderOrders();
  } catch {
    els.orderPanel.innerHTML =
      '<div class="empty-hint">订单数据加载失败，请确认服务已启动</div>';
  }
}

function renderOrders(filter = "") {
  const kw = filter.trim().toUpperCase();
  const rows = state.orders.filter(
    (o) =>
      !kw ||
      o.order_id.toUpperCase().includes(kw) ||
      o.status.includes(kw) ||
      o.items.toUpperCase().includes(kw)
  );

  els.orderList.innerHTML = "";

  if (!rows.length) {
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.textContent = state.orders.length ? "没有匹配的订单" : "暂无订单数据";
    els.orderList.appendChild(hint);
    return;
  }

  for (const o of rows) {
    const btn = document.createElement("button");
    btn.className = "order-item";
    btn.innerHTML = `
      <div class="row-1">
        <span class="oid">${escapeHtml(o.order_id)}</span>
        <span class="status-pill" data-s="${escapeHtml(o.status)}">${escapeHtml(o.status)}</span>
      </div>
      <div class="row-1">
        <span class="meta"></span>
        <span class="amount">¥${escapeHtml(o.amount)}</span>
      </div>`;
    btn.querySelector(".meta").textContent = o.items;
    btn.title = `${o.carrier} ${o.tracking_no}`;
    btn.addEventListener("click", () => {
      switchTab("chats");
      els.input.value = `帮我查一下订单 ${o.order_id} 的进度`;
      autoResize();
      updateOrderChip();
      els.input.focus();
    });
    els.orderList.appendChild(btn);
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  els.chatPanel.classList.toggle("hidden", name !== "chats");
  els.orderPanel.classList.toggle("hidden", name !== "orders");
  if (name === "orders" && !state.orders.length) loadOrders();
}

/* ------------------------------------------------------------ 健康检查 */

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.agent_ready) {
      els.healthDot.className = "dot on";
      els.healthText.textContent = "服务正常 · Agent 已就绪";
      return true;
    }
    els.healthDot.className = "dot warn";
    els.healthText.textContent = "Agent 加载中…";
    return false;
  } catch {
    els.healthDot.className = "dot off";
    els.healthText.textContent = "无法连接后端服务";
    return false;
  }
}

async function watchHealth() {
  let tries = 0;
  while (tries < 40) {
    const ok = await checkHealth();
    if (ok) break;
    tries += 1;
    await new Promise((r) => setTimeout(r, 1500));
  }
}

/* ------------------------------------------------------------ 事件绑定 */

function bindEvents() {
  els.send.addEventListener("click", () => sendMessage());

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  els.input.addEventListener("input", () => {
    autoResize();
    updateOrderChip();
  });

  els.newChat.addEventListener("click", () => {
    if (state.streaming) return;
    newThread();
    els.input.focus();
  });

  els.clearChat.addEventListener("click", async () => {
    if (state.streaming) return;
    const chat = currentChat();
    if (!chat) return;
    chat.messages = [];
    chat.title = "新会话";
    renderChatList();
    renderThread();
    saveStore();
    try {
      await fetch(`/api/reset?thread_id=${encodeURIComponent(chat.id)}`, { method: "POST" });
    } catch {
      /* 记忆清理失败不影响界面 */
    }
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  els.orderSearch.addEventListener("input", (e) => renderOrders(e.target.value));

  document.querySelectorAll(".suggest").forEach((btn) => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.q || btn.textContent));
  });
}

/* ------------------------------------------------------------ 启动 */

function boot() {
  loadStore();
  bindEvents();

  if (!state.chats.length) {
    state.chats.push({
      id: "t-" + Date.now().toString(36),
      title: "新会话",
      updatedAt: Date.now(),
      messages: [],
    });
  }
  state.threadId = state.chats[0].id;

  renderChatList();
  renderThread();
  updateRouteBadge("");
  autoResize();
  watchHealth();
  els.input.focus();
}

boot();
