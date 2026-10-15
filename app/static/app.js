/* Atlas front-end: a tiny AG-UI client. It POSTs RunAgentInput to /agent and renders the SSE event stream. */
const $ = (s, r = document) => r.querySelector(s);
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));

const el = {
  messages: $("#messages"), empty: $("#empty"), input: $("#input"), send: $("#send"), form: $("#composer"),
  steps: $("#steps"), route: $("#route"), traceEmpty: $("#trace-empty"),
  sourcesWrap: $("#sources-wrap"), sources: $("#sources"),
  docList: $("#doc-list"), file: $("#file"), drop: $("#drop"), status: $("#upload-status"),
  mode: $("#mode"), modeHint: $("#mode-hint"), health: $("#health"),
};

const MODE_HINTS = {
  auto: "Atlas decides whether to use your files, the web, or neither.",
  docs: "Answers come only from the files you have added.",
  web: "Answers come only from a live web search.",
  hybrid: "Atlas checks your files and the web, then combines them.",
};

let threadId = localStorage.getItem("atlas.thread") || uid();
localStorage.setItem("atlas.thread", threadId);
let mode = "auto";
let busy = false;

/* ───────── markdown ───────── */
function renderMarkdown(text) {
  if (!window.marked || !window.DOMPurify) return null;
  const html = DOMPurify.sanitize(marked.parse(text, { breaks: true }));
  return html.replace(/\[(\d{1,2})\]/g, '<button type="button" class="cite" data-n="$1">$1</button>');
}

/* ───────── chat helpers ───────── */
function addUser(text) {
  el.empty?.remove();
  const d = document.createElement("div");
  d.className = "msg user";
  d.textContent = text;
  el.messages.append(d);
  scrollDown();
}
function addBot() {
  el.empty?.remove();
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  const body = document.createElement("div");
  body.className = "body cursor";
  wrap.append(body);
  el.messages.append(wrap);
  return { wrap, body, text: "", raf: 0 };
}
function paint(bot) {
  bot.raf = 0;
  const html = renderMarkdown(bot.text);
  if (html === null) bot.body.textContent = bot.text; else bot.body.innerHTML = html;
  scrollDown();
}
function scrollDown() { el.messages.scrollTop = el.messages.scrollHeight; }

/* ───────── trace panel ───────── */
function resetTrace() {
  el.steps.innerHTML = ""; el.route.hidden = true; el.sources.innerHTML = ""; el.sourcesWrap.hidden = true;
  el.traceEmpty.hidden = true;
}
const stepEls = {};
function startStep(ev) {
  const li = document.createElement("li");
  li.className = "step running";
  li.innerHTML = `<span class="label"></span><span class="ms"></span><div class="note"></div>`;
  $(".label", li).textContent = ev.label || ev.stepName;
  el.steps.append(li);
  stepEls[ev.stepName] = li;
}
function finishStep(ev, snap) {
  const li = stepEls[ev.stepName];
  if (!li) return;
  li.classList.replace("running", "done");
  $(".ms", li).textContent = ev.durationMs >= 1000 ? (ev.durationMs / 1000).toFixed(1) + " s" : ev.durationMs + " ms";
  if (ev.stepName === "grade" && snap?.grade) {
    $(".note", li).textContent = (snap.grade === "sufficient" ? "Enough evidence. " : "Not enough evidence. ") + (snap.gradeReason || "");
  }
  if (ev.stepName === "rewrite_query" && snap?.searchQuery) $(".note", li).textContent = "New query: " + snap.searchQuery;
}
const toolEls = {};
function toolStart(ev) {
  const holder = el.steps.lastElementChild; if (!holder) return;
  const d = document.createElement("div");
  d.className = "tool";
  d.innerHTML = `<b></b><code></code><div class="res"></div>`;
  $("b", d).textContent = ev.toolCallName;
  holder.append(d);
  toolEls[ev.toolCallId] = { d, args: "" };
}
function toolArgs(ev) {
  const t = toolEls[ev.toolCallId]; if (!t) return;
  t.args += ev.delta;
  try { $("code", t.d).textContent = JSON.stringify(JSON.parse(t.args)); } catch { $("code", t.d).textContent = t.args; }
}
function toolResult(ev) {
  const t = toolEls[ev.toolCallId]; if (!t) return;
  let r; try { r = JSON.parse(ev.content); } catch { r = {}; }
  const out = $(".res", t.d);
  if (r.error) out.textContent = "Failed: " + r.error;
  else if (r.matches !== undefined) out.textContent = r.matches ? `${r.matches} passages found (best match ${r.top_score})` : "No matching passages";
  else if (r.results) out.textContent = r.results.length ? `${r.results.length} web results` : "No web results";
}
function showRoute(v) {
  const labels = { direct: "Answer directly", docs: "Use your files", web: "Search the web", hybrid: "Files and web" };
  el.route.hidden = false;
  el.route.innerHTML = `<b></b><span></span>`;
  $("b", el.route).textContent = labels[v.route] || v.route;
  $("span", el.route).textContent = v.reasoning || "";
}
function showSources(list) {
  el.sources.innerHTML = "";
  el.sourcesWrap.hidden = !list.length;
  for (const s of list) {
    const li = document.createElement("li");
    li.id = "src-" + s.id;
    const title = s.url ? `<a href="${encodeURI(s.url)}" target="_blank" rel="noopener"></a>` : `<span></span>`;
    li.innerHTML = `<div class="t"><span class="n">${s.id}</span>${title}</div><div class="kind"></div><p></p>`;
    $(".t a, .t span:last-child", li).textContent = s.title;
    $(".kind", li).textContent = (s.kind === "web" ? "Web" : "Your file") + " · relevance " + s.score;
    $("p", li).textContent = s.snippet;
    el.sources.append(li);
  }
}
function showFollowups(bot, qs) {
  const d = document.createElement("div");
  d.className = "followups";
  for (const q of qs) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = q; b.onclick = () => ask(q);
    d.append(b);
  }
  bot.wrap.append(d);
  scrollDown();
}

/* ───────── AG-UI run ───────── */
async function ask(text) {
  text = (text || "").trim();
  if (!text || busy) return;
  busy = true; el.send.disabled = true;
  el.input.value = ""; el.input.style.height = "auto";
  addUser(text);
  resetTrace();

  const bot = addBot();
  let snap = null, gotText = false, failed = false;

  const onEvent = (ev) => {
    switch (ev.type) {
      case "STEP_STARTED": startStep(ev); break;
      case "STATE_SNAPSHOT": snap = ev.snapshot; break;
      case "STEP_FINISHED": finishStep(ev, snap); break;
      case "TOOL_CALL_START": toolStart(ev); break;
      case "TOOL_CALL_ARGS": toolArgs(ev); break;
      case "TOOL_CALL_RESULT": toolResult(ev); break;
      case "TEXT_MESSAGE_CONTENT":
        gotText = true; bot.text += ev.delta;
        if (!bot.raf) bot.raf = requestAnimationFrame(() => paint(bot));
        break;
      case "CUSTOM":
        if (ev.name === "route") showRoute(ev.value);
        else if (ev.name === "sources") showSources(ev.value);
        else if (ev.name === "followups") showFollowups(bot, ev.value);
        break;
      case "RUN_ERROR":
        failed = true;
        bot.body.className = "error"; bot.body.textContent = ev.message;
        break;
    }
  };

  try {
    const res = await fetch("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        threadId, runId: uid(),
        messages: [{ id: uid(), role: "user", content: text }],
        state: { mode }, forwardedProps: {}, tools: [], context: [],
      }),
    });
    if (!res.ok || !res.body) throw new Error("Server returned " + res.status);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, i); buf = buf.slice(i + 2);
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (line) { try { onEvent(JSON.parse(line.slice(5))); } catch (e) { console.warn("bad frame", e); } }
      }
    }
  } catch (e) {
    failed = true;
    bot.body.className = "error";
    bot.body.textContent = "Could not reach the agent: " + e.message;
  } finally {
    if (bot.raf) cancelAnimationFrame(bot.raf);
    if (gotText) paint(bot);
    if (!failed) bot.body.classList.remove("cursor");
    busy = false; el.send.disabled = false; el.input.focus();
  }
}

/* ───────── events ───────── */
el.form.addEventListener("submit", (e) => { e.preventDefault(); ask(el.input.value); });
el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(el.input.value); }
});
el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 160) + "px";
});
document.addEventListener("click", (e) => {
  const sug = e.target.closest(".suggest button");
  if (sug) return ask(sug.dataset.q);
  const cite = e.target.closest(".cite");
  if (cite) {
    const li = document.getElementById("src-" + cite.dataset.n);
    if (li) { li.scrollIntoView({ block: "nearest", behavior: "smooth" }); li.classList.add("flash"); setTimeout(() => li.classList.remove("flash"), 1400); }
  }
});
$("#new-chat").addEventListener("click", () => {
  threadId = uid(); localStorage.setItem("atlas.thread", threadId);
  location.reload();
});
el.mode.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-mode]"); if (!b) return;
  mode = b.dataset.mode;
  el.mode.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  el.modeHint.textContent = MODE_HINTS[mode];
});

/* ───────── documents ───────── */
async function loadDocs() {
  const docs = await (await fetch("/api/documents")).json();
  el.docList.innerHTML = "";
  for (const d of docs) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><small></small><button type="button" aria-label="Remove file">Remove</button>`;
    $(".name", li).textContent = d.name; $(".name", li).title = d.name;
    $("small", li).textContent = d.chunks + " parts";
    $("button", li).onclick = async () => { await fetch("/api/documents/" + encodeURIComponent(d.name), { method: "DELETE" }); loadDocs(); };
    el.docList.append(li);
  }
}
async function upload(file) {
  if (!file) return;
  el.status.textContent = `Adding ${file.name}…`;
  const fd = new FormData(); fd.append("file", file);
  const res = await fetch("/api/documents", { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  el.status.textContent = res.ok ? `Added ${data.name} (${data.chunks} parts).` : (data.detail || "Upload failed.");
  el.file.value = ""; loadDocs();
}
el.file.addEventListener("change", () => upload(el.file.files[0]));
["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
el.drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

(async () => {
  loadDocs();
  try {
    const h = await (await fetch("/api/health")).json();
    if (!h.ok) { el.health.hidden = false; el.health.textContent = "Add " + h.missing_keys.join(", ") + " to your .env file, then restart the server."; }
  } catch { /* ignore */ }
})();
