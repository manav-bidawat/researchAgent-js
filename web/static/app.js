/* Front end for the demo server.
   In:  /api/papers for the corpus view, /api/ask as an SSE stream of loop events.
   Out: the answer rendered from markdown with its citations turned into controls, and
        the agent's steps drawn in the margin rail as they arrive. No libraries. */

const $ = (sel) => document.querySelector(sel);

/* ------------------------------------------------------------------ markdown */

const esc = (s) => s.replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* A chunk id is "<paper_id>__c0007". The agent writes them inline, several in a row and
   often several from one paper, which is unreadable as raw ids. They are numbered per
   answer in order of first appearance, the way a paper numbers its references, and each
   number stays clickable. Handled before any other inline rule so an id with underscores
   never reaches the emphasis pass. */
const CITE = /\[([A-Za-z0-9._+-]+__[cf]\d+)\]/g;
const CITE_RUN = /(?:\[[A-Za-z0-9._+-]+__[cf]\d+\]\s*){2,}/g;

let refs = new Map();   // paper_id -> { n, chunk }

function refNumber(chunkId) {
  const paper = chunkId.split("__")[0];
  if (!refs.has(paper)) refs.set(paper, { n: refs.size + 1, chunk: chunkId });
  return refs.get(paper).n;
}

function inline(text) {
  // "[a__c1][a__c9][b__c2]" cites two papers, not three: one marker per paper per run.
  let src = String(text).replace(CITE_RUN, (run) => {
    const seen = new Set();
    const kept = [];
    (run.match(CITE) || []).forEach((token) => {
      const paper = token.slice(1, -1).split("__")[0];
      if (!seen.has(paper)) { seen.add(paper); kept.push(token); }
    });
    return kept.join("");
  });

  let out = esc(src);
  out = out.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  out = out.replace(CITE, (_, id) =>
    `<button class="cite" data-paper="${id.split("__")[0]}" data-chunk="${id}" ` +
    `title="Show this paper">${refNumber(id)}</button>`);
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return out;
}

const cells = (row) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|");

/* Handles what the agent actually emits: headings, lists, tables, quotes, fenced code,
   paragraphs. Deliberately small — this is not a general markdown implementation. */
function markdown(src) {
  const lines = String(src || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let list = null;     // "ul" | "ol" | null
  let para = [];

  const flushPara = () => {
    if (para.length) { html.push(`<p>${inline(para.join(" "))}</p>`); para = []; }
  };
  const flushList = () => { if (list) { html.push(`</${list}>`); list = null; } };
  const flush = () => { flushPara(); flushList(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      flush();
      const body = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i]);
      html.push(`<pre><code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }

    if (!line.trim()) { flush(); continue; }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); html.push("<hr>"); continue; }

    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
      flush();
      const head = cells(line).map((c) => `<th>${inline(c.trim())}</th>`).join("");
      const body = [];
      i += 1;
      while (/^\s*\|.*\|\s*$/.test(lines[i + 1] || "")) {
        i += 1;
        body.push("<tr>" + cells(lines[i]).map((c) => `<td>${inline(c.trim())}</td>`).join("") + "</tr>");
      }
      html.push(`<table><thead><tr>${head}</tr></thead><tbody>${body.join("")}</tbody></table>`);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flush();
      html.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const number = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || number) {
      flushPara();
      const want = bullet ? "ul" : "ol";
      if (list !== want) { flushList(); html.push(`<${want}>`); list = want; }
      html.push(`<li>${inline((bullet || number)[1])}</li>`);
      continue;
    }

    flushList();
    para.push(line.trim());
  }

  flush();
  return html.join("\n");
}

/* The numbers are only meaningful if you can read them off somewhere. */
function references() {
  if (!refs.size) return "";
  const rows = [...refs.entries()].map(([paper, ref]) => {
    const record = papersById.get(paper);
    const title = record ? esc(record.title) : esc(paper);
    const line = record ? `<span class="ref-who">${esc(who(record))}</span>` : "";
    const link = record && record.pdf_url
      ? ` <a href="${esc(record.pdf_url)}" target="_blank" rel="noopener">arXiv</a>` : "";
    return `<li value="${ref.n}"><span class="ref-title">${title}</span> ${line}${link}</li>`;
  });
  return `<ol class="refs">${rows.join("")}</ol>`;
}

/* ------------------------------------------------------------------ corpus */

const papersById = new Map();

const count = (n, noun) => `${n} ${noun}${n === 1 ? "" : "s"}`;

const who = (p) => {
  const names = p.authors || [];
  const lead = names.length > 2 ? `${names[0]} and ${names.length - 1} others` : names.join(" and ");
  return [lead, p.year].filter(Boolean).join(", ");
};

async function loadCorpus() {
  let data;
  try {
    data = await (await fetch("/api/papers")).json();
  } catch (err) {
    $("#corpus-scope").textContent = "The server is not answering. Is it still running?";
    return;
  }

  (data.papers || []).forEach((p) => papersById.set(p.paper_id, p));

  const n = (data.papers || []).length;
  if (!n) {
    $("#corpus-scope").textContent = data.error ||
      'Nothing indexed yet. Ask a question, or run npx tsx app/cli.ts index "your topic".';
    // Not a dead end: search_literature can collect a corpus from nothing mid-question,
    // which is exactly what examples/cold_start.md demonstrates.
    $("#scope").textContent = "Nothing indexed yet — ask anyway, and the agent will go and find the papers first.";
    return;
  }

  const topics = data.topics || [];
  $("#corpus-scope").textContent =
    `${n} papers, collected under ${topics.length} ${topics.length === 1 ? "topic" : "topics"}. ` +
    `Embedded with ${data.embedding_model || "the configured model"}.`;
  // Deliberately says nothing about a paper count. The corpus is not a fixed boundary:
  // the agent can call search_literature and grow it while answering, so a line promising
  // "only these N papers" would be wrong the moment it did.
  $("#scope").textContent = "Every claim is cited, and each citation opens the paper behind it. If the corpus does not cover your question, the agent searches arXiv and reads more.";

  const box = $("#corpus");
  box.innerHTML = "";

  // A paper tagged with something that is not a key of manifest.topics would belong to
  // no section and quietly vanish, leaving a count that does not match the list. Give
  // the leftovers a section of their own instead.
  const known = new Set(topics.map((t) => t.tag));
  const loose = (data.papers || []).filter(
    (p) => !(p.topic_tags || []).some((tag) => known.has(tag)));
  const sections = loose.length
    ? [...topics, { tag: null, label: "Collected under no current topic", arxiv_query: "" }]
    : topics;

  sections.forEach((topic) => {
    const section = document.createElement("section");
    section.className = "topic";
    const mine = topic.tag === null
      ? loose
      : (data.papers || []).filter((p) => (p.topic_tags || []).includes(topic.tag));
    section.innerHTML =
      `<h2 class="topic-name">${esc(topic.label)}</h2>` +
      (topic.arxiv_query ? `<p class="topic-query">${esc(topic.arxiv_query)}</p>` : "") +
      mine.map((p) => `
        <div class="paper">
          <div class="paper-title">${esc(p.title)}</div>
          <div class="paper-who">${esc(who(p))}</div>
          <span class="paper-file">${esc(p.pdf_name || p.paper_id)}${
            p.n_chunks ? ` — ${count(p.n_chunks, "chunk")}, ${count(p.n_figures, "figure")}` : ""
          }${p.parse_status && p.parse_status !== "ok"
              ? ` <span class="paper-flag">(${esc(p.parse_status)} parse)</span>` : ""
          }${p.pdf_url ? ` <a href="${esc(p.pdf_url)}" target="_blank" rel="noopener">arXiv</a>` : ""}</span>
        </div>`).join("");
    box.appendChild(section);
  });
}

/* ------------------------------------------------------------------ rail */

const rail = $("#rail");
let liveStep = null;

function railClear() {
  rail.innerHTML = "";
  liveStep = null;
}

function railRow(cls, tool, name, arg, out) {
  const row = document.createElement("div");
  row.className = `step ${cls}`;
  if (tool) row.dataset.tool = tool;
  row.innerHTML =
    `<span class="dot"></span><span><span class="step-name">${esc(name)}</span>` +
    (arg ? `<span class="step-arg">${esc(arg)}</span>` : "") +
    `<span class="step-out">${esc(out || "")}</span></span>`;
  rail.appendChild(row);
  return row;
}

/* One short line per call. check_evidence_consistency is handed the whole draft answer,
   which would otherwise fill the rail, so every value is clipped, not just the line. */
const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

const argLine = (args) => Object.entries(args || {})
  .map(([k, v]) => `${k}=${clip(JSON.stringify(v), 54)}`)
  .join("  ");

/* Each tool returns a different dict, so read whichever count field it has. */
function outcome(result) {
  if (!result || typeof result !== "object") return "done";
  if (result.error) return `failed: ${result.error}`;
  if (Array.isArray(result.chunks)) {
    const n = result.chunks.length;
    return `${n} ${n === 1 ? "passage" : "passages"}` +
           (result.sufficient_evidence === false ? ", not enough to answer" : "");
  }
  if (Array.isArray(result.papers_added))
    return `${result.papers_added.length} papers added, ${result.chunks_added || 0} chunks`;
  if (Array.isArray(result.claims))
    return `${result.claims.length} claims checked, ${Math.round((result.grounded_ratio || 0) * 100)}% grounded`;
  if ("found" in result)
    return result.found ? "a disagreement to look at" : "no disagreement found";
  if (result.summary) return String(result.summary).slice(0, 90);
  if (result.image_path) return "figure opened";
  return "done";
}

const secs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);

function onStep(ev) {
  if (ev.kind === "thinking") {
    liveStep = railRow("is-thinking is-live", null, "thinking", "", "");
    return;
  }
  if (ev.kind === "answering") {
    if (liveStep) liveStep.remove();
    liveStep = railRow("is-thinking", null, "writing the answer", "", "");
    return;
  }
  if (ev.kind === "tool_start") {
    if (liveStep) liveStep.remove();
    liveStep = railRow("is-live", ev.tool_name, ev.tool_name, argLine(ev.args), "working");
    return;
  }
  if (ev.kind === "tool_end") {
    if (liveStep) {
      liveStep.classList.remove("is-live");
      liveStep.querySelector(".step-out").textContent =
        `${outcome(ev.result)} — ${secs(ev.latency_ms || 0)}`;
      liveStep = null;
    }
    // search_literature indexes new papers mid-question, so the corpus this page
    // described when it loaded is now out of date — including the paper the answer is
    // about to cite. Start refetching now; the agent still has a turn or two to go.
    if ((ev.result?.papers_added || []).length) {
      corpusStale = true;
      corpusReady = loadCorpus();
    }
    return;
  }
  if (ev.kind === "skipped") {
    if (liveStep) { liveStep.remove(); liveStep = null; }
    railRow("", ev.tool_name, ev.tool_name || "call", "", `skipped: ${ev.reason}`);
    return;
  }
  if (ev.kind === "image") {
    railRow("", "inspect_figure", "looking at the figure", "", "");
    return;
  }
  if (ev.kind === "cap") {
    if (liveStep) { liveStep.remove(); liveStep = null; }
    railRow("", null, "out of steps", "", "answering with what it has");
  }
}

/* ------------------------------------------------------------------ ask */

const answer = $("#answer");
let stream = null;
let corpusStale = false;
let corpusReady = null;

let lastAnswer = null;

/* Kept separate from the stream handler so it can run again once a mid-question
   search_literature has landed and papersById knows the new papers. */
function renderAnswer() {
  const data = lastAnswer;
  if (!data) return;
  refs = new Map();
  answer.innerHTML = markdown(data.answer) + references() +
    `<p class="verdict">${data.tool_calls} tool ${data.tool_calls === 1 ? "call" : "calls"} ` +
    `over ${data.iterations} ${data.iterations === 1 ? "turn" : "turns"}, ` +
    `${data.context_tokens.toLocaleString()} tokens of context. ` +
    `Trace saved as ${data.run_id}.</p>`;
}

function ask(question) {
  if (stream) stream.close();
  lastAnswer = null;
  corpusStale = false;
  corpusReady = null;
  railClear();
  answer.innerHTML = '<p class="blank">Working.</p>';
  $("#go").disabled = true;

  stream = new EventSource(`/api/ask?q=${encodeURIComponent(question)}`);

  stream.addEventListener("step", (e) => onStep(JSON.parse(e.data)));

  stream.addEventListener("answer", (e) => {
    lastAnswer = JSON.parse(e.data);
    renderAnswer();
  });

  stream.addEventListener("failed", (e) => {
    const data = JSON.parse(e.data);
    refs = new Map();
    answer.innerHTML =
      (data.answer ? markdown(data.answer) : "") +
      `<p class="failure">${esc(data.error)} — ${esc(data.detail || "")}</p>`;
  });

  // The server closes the connection when it is finished, and EventSource would treat
  // that as a dropped link and re-run the whole question. Close it from this side first.
  stream.addEventListener("done", async () => {
    stream.close();
    stream = null;
    $("#go").disabled = false;
    if (liveStep && liveStep.classList.contains("is-thinking")) liveStep.remove();
    if (corpusStale) {
      await (corpusReady || loadCorpus());
      corpusStale = false;
      renderAnswer();   // the new papers can now be named, not printed as bare ids
    }
  });

  stream.onerror = () => {
    if (!stream) return;
    stream.close();
    stream = null;
    $("#go").disabled = false;
    answer.innerHTML = '<p class="failure">Lost the connection to the server.</p>';
  };
}

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const question = $("#q").value.trim() || $("#q").placeholder;
  $("#q").value = question;
  ask(question);
});

/* ------------------------------------------------------------------ citation card */

const card = $("#card");

function hideCard() { card.classList.add("is-hidden"); }

document.addEventListener("click", (e) => {
  const chip = e.target.closest(".cite");
  if (!chip) { hideCard(); return; }

  const paper = papersById.get(chip.dataset.paper);
  card.innerHTML = paper
    ? `<div class="card-title">${esc(paper.title)}</div>` +
      `<div class="card-who">${esc(who(paper))}</div>` +
      `<span class="card-id">${esc(chip.dataset.chunk)}` +
      (paper.pdf_url ? ` — <a href="${esc(paper.pdf_url)}" target="_blank" rel="noopener">open on arXiv</a>` : "") +
      `</span>`
    : `<div class="card-title">Not in the corpus</div>` +
      `<span class="card-id">${esc(chip.dataset.chunk)}</span>`;

  const box = chip.getBoundingClientRect();
  card.classList.remove("is-hidden");
  const width = card.offsetWidth;
  card.style.left = `${Math.max(12, Math.min(box.left, window.innerWidth - width - 12))}px`;
  card.style.top = `${box.bottom + 8}px`;
});

document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideCard(); });

/* ------------------------------------------------------------------ views */

document.querySelectorAll(".view-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach((t) => t.classList.toggle("is-on", t === tab));
    $("#view-ask").classList.toggle("is-hidden", tab.dataset.view !== "ask");
    $("#view-corpus").classList.toggle("is-hidden", tab.dataset.view !== "corpus");
    hideCard();
  });
});

loadCorpus();
