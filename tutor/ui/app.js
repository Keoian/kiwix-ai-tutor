"use strict";

(function () {
  const chat = document.getElementById("chat");
  const input = document.getElementById("input");
  const composer = document.getElementById("composer");
  const sendBtn = document.getElementById("send");
  const stopBtn = document.getElementById("stop");
  const subjectSelect = document.getElementById("subject");
  const sourceViewerBody = document.getElementById("source-viewer-body");
  const statusPanelBody = document.getElementById("status-panel-body");
  const actionButtons = document.querySelectorAll("#action-bar button[data-action]");

  let sessionId = null;
  let currentAbortController = null;

  // ---------------------------------------------------------------------
  // DOM helpers (never assign markup as a page fragment: build nodes explicitly)
  // ---------------------------------------------------------------------

  function el(tag, opts) {
    const node = document.createElement(tag);
    opts = opts || {};
    if (opts.className) node.className = opts.className;
    if (opts.text !== undefined) node.textContent = opts.text;
    if (opts.attrs) {
      for (const key in opts.attrs) {
        node.setAttribute(key, opts.attrs[key]);
      }
    }
    return node;
  }

  function appendMessage(kind, text) {
    const wrapper = el("div", { className: "msg msg-" + kind });
    renderTextWithCitations(wrapper, text || "");
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
    return wrapper;
  }

  // Splits text on [S1], [S2, S3] style citation labels and renders each
  // label group as a tappable chip; everything else stays as plain text
  // nodes via textContent only, no raw markup assignment.
  const CITATION_RE = /\[(S\d+(?:\s*,\s*S\d+)*)\]/g;

  function renderTextWithCitations(container, text) {
    let lastIndex = 0;
    let match;
    CITATION_RE.lastIndex = 0;
    while ((match = CITATION_RE.exec(text)) !== null) {
      if (match.index > lastIndex) {
        container.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      const labels = match[1].split(",").map(function (s) { return s.trim(); });
      labels.forEach(function (label) {
        container.appendChild(renderCitationChip(label));
      });
      lastIndex = CITATION_RE.lastIndex;
    }
    if (lastIndex < text.length) {
      container.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function renderCitations(container, citations) {
    (citations || []).forEach(function (citation) {
      const label = citation.label || citation;
      container.appendChild(renderCitationChip(label, citation.passage_id));
    });
  }

  function renderCitationChip(label, passageId) {
    const chip = el("button", {
      className: "citation-chip",
      text: "[" + label + "]",
      attrs: { type: "button", "data-passage-id": passageId || label },
    });
    chip.addEventListener("click", function () {
      openSourceViewer(chip.getAttribute("data-passage-id"));
    });
    return chip;
  }

  function appendCalcResult(text) {
    const wrapper = el("div", { className: "msg msg-tutor" });
    const badge = el("span", { className: "calc-result", text: text });
    wrapper.appendChild(badge);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
  }

  function appendError(message) {
    appendMessage("error", message);
  }

  // ---------------------------------------------------------------------
  // Session
  // ---------------------------------------------------------------------

  async function ensureSession() {
    if (sessionId) return sessionId;
    const resp = await fetch("/api/session", { method: "POST" });
    const body = await resp.json();
    sessionId = body.session_id;
    return sessionId;
  }

  subjectSelect.addEventListener("change", async function () {
    const sid = await ensureSession();
    await fetch("/api/session/" + sid + "/subject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject: subjectSelect.value }),
    });
  });

  // ---------------------------------------------------------------------
  // Turn streaming: SSE over a POST response, so we cannot use
  // EventSource (GET-only). Instead: fetch() + ReadableStream, decoding
  // and parsing "event: name\ndata: {...}\n\n" frames ourselves.
  // ---------------------------------------------------------------------

  async function sendTurn(payload) {
    const sid = await ensureSession();
    setBusy(true);

    let tutorLine = "";
    const tutorNode = appendMessage("tutor", "");

    currentAbortController = new AbortController();
    try {
      const resp = await fetch("/api/session/" + sid + "/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: currentAbortController.signal,
      });

      if (!resp.ok) {
        appendError("Request failed (" + resp.status + ")");
        setBusy(false);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        let sepIndex;
        while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);
          handleFrame(frame, tutorNode, function (text) {
            tutorLine = text;
          }, function () {
            return tutorLine;
          });
        }
      }
    } catch (err) {
      appendError("Connection lost.");
    } finally {
      setBusy(false);
      currentAbortController = null;
    }
  }

  function handleFrame(frame, tutorNode, setTutorLine, getTutorLine) {
    let eventName = "message";
    let dataText = "";
    frame.split("\n").forEach(function (line) {
      if (line.indexOf("event:") === 0) {
        eventName = line.slice(6).trim();
      } else if (line.indexOf("data:") === 0) {
        dataText += line.slice(5).trim();
      }
    });

    let data = {};
    try {
      data = dataText ? JSON.parse(dataText) : {};
    } catch (err) {
      data = {};
    }

    if (eventName === "token") {
      const text = getTutorLine() + (data.text || "");
      setTutorLine(text);
      tutorNode.textContent = "";
      renderTextWithCitations(tutorNode, text);
    } else if (eventName === "tool") {
      if (data.kind === "calc" || data.name === "calc") {
        appendCalcResult(data.result !== undefined ? String(data.result) : JSON.stringify(data));
      }
    } else if (eventName === "citations") {
      renderCitations(tutorNode, data.citations);
    } else if (eventName === "error") {
      appendError(data.message || "An error occurred.");
    } else if (eventName === "done") {
      // stream complete; nothing further to render here.
    }
  }

  function setBusy(isBusy) {
    sendBtn.disabled = isBusy;
    stopBtn.disabled = !isBusy;
    actionButtons.forEach(function (btn) {
      btn.disabled = isBusy;
    });
  }

  composer.addEventListener("submit", function (evt) {
    evt.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    appendMessage("user", text);
    input.value = "";
    sendTurn({ text: text });
  });

  actionButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const action = btn.getAttribute("data-action");
      appendMessage("user", "[" + action + "]");
      sendTurn({ action: action });
    });
  });

  stopBtn.addEventListener("click", async function () {
    if (!sessionId) return;
    await fetch("/api/session/" + sessionId + "/stop", { method: "POST" });
    if (currentAbortController) {
      currentAbortController.abort();
    }
  });

  // ---------------------------------------------------------------------
  // Source viewer
  // ---------------------------------------------------------------------

  async function openSourceViewer(passageId) {
    sourceViewerBody.textContent = "";
    sourceViewerBody.appendChild(el("p", { text: "Loading..." }));
    try {
      const resp = await fetch("/api/source/" + encodeURIComponent(passageId));
      if (!resp.ok) {
        sourceViewerBody.textContent = "";
        sourceViewerBody.appendChild(el("p", { text: "Source not found." }));
        return;
      }
      const body = await resp.json();
      renderSourceView(body);
    } catch (err) {
      sourceViewerBody.textContent = "";
      sourceViewerBody.appendChild(el("p", { text: "Could not load source." }));
    }
  }

  function renderSourceView(body) {
    sourceViewerBody.textContent = "";

    const heading = el("h3", { text: body.title || "" });
    sourceViewerBody.appendChild(heading);

    if (body.heading_path && body.heading_path.length) {
      sourceViewerBody.appendChild(el("p", { className: "source-path", text: body.heading_path.join(" > ") }));
    }

    const passage = el("p");
    const start = body.highlight ? body.highlight.start : 0;
    const end = body.highlight ? body.highlight.end : 0;

    passage.appendChild(document.createTextNode(body.context_before || ""));

    const highlightSpan = el("span", {
      className: "source-highlight",
      text: (body.text || "").slice(start, end),
    });
    passage.appendChild(highlightSpan);

    passage.appendChild(document.createTextNode(body.context_after || ""));
    sourceViewerBody.appendChild(passage);
  }

  // ---------------------------------------------------------------------
  // Status panel
  // ---------------------------------------------------------------------

  async function refreshStatus() {
    try {
      const url = sessionId ? "/api/status?session_id=" + encodeURIComponent(sessionId) : "/api/status";
      const resp = await fetch(url);
      if (!resp.ok) return;
      const body = await resp.json();
      renderStatus(body);
    } catch (err) {
      // Offline/status errors are non-fatal for the chat itself.
    }
  }

  function renderStatus(body) {
    statusPanelBody.textContent = "";
    const rows = [
      ["LLM", body.llm && body.llm.healthy ? "healthy" : "unavailable"],
      ["Model", body.model_path || ""],
      ["Context ceiling", body.context_ceiling != null ? String(body.context_ceiling) : ""],
    ];
    if (body.tokens_used != null) {
      rows.push(["Tokens used", String(body.tokens_used)]);
    }
    if (body.headroom != null) {
      rows.push(["Headroom", String(body.headroom)]);
    }
    rows.forEach(function (pair) {
      statusPanelBody.appendChild(el("dt", { text: pair[0] }));
      statusPanelBody.appendChild(el("dd", { text: pair[1] }));
    });
  }

  ensureSession().then(refreshStatus);
  setInterval(refreshStatus, 15000);
})();
