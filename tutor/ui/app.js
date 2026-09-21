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
  const studentSelect = document.getElementById("student-select");
  const newStudentBtn = document.getElementById("new-student-btn");
  const newStudentForm = document.getElementById("new-student-form");
  const newStudentName = document.getElementById("new-student-name");
  const newStudentGrade = document.getElementById("new-student-grade");
  const newLessonBtn = document.getElementById("new-lesson-btn");
  const lessonResumeList = document.getElementById("lesson-resume-list");

  let sessionId = null;
  let currentAbortController = null;
  let currentStudentId = null;
  let lastTurnMeta = null;
  let lastCitationsEvent = null;
  let lastAttributionsEvent = null;

  // ---------------------------------------------------------------------
  // Students / lessons (WP-C4: profile selector, new lesson, resume list)
  // ---------------------------------------------------------------------

  async function refreshStudents() {
    try {
      const resp = await fetch("/api/profiles");
      if (!resp.ok) return;
      const body = await resp.json();
      studentSelect.textContent = "";
      (body.profiles || []).forEach(function (profile) {
        const opt = el("option", { text: profile.display_name });
        opt.value = profile.id;
        studentSelect.appendChild(opt);
      });
      if (body.profiles && body.profiles.length && !currentStudentId) {
        currentStudentId = body.profiles[0].id;
        studentSelect.value = currentStudentId;
        refreshLessons();
      }
    } catch (err) {
      // Offline/profiles-not-configured is non-fatal for the chat itself.
    }
  }

  async function refreshLessons() {
    if (!currentStudentId) return;
    try {
      const resp = await fetch("/api/profiles/" + encodeURIComponent(currentStudentId) + "/lessons");
      if (!resp.ok) return;
      const body = await resp.json();
      lessonResumeList.textContent = "";
      (body.lessons || []).forEach(function (lesson) {
        const opt = el("option", {
          text: lesson.subject + (lesson.ended ? " (ended)" : ""),
        });
        opt.value = lesson.id;
        lessonResumeList.appendChild(opt);
      });
    } catch (err) {
      // non-fatal
    }
  }

  studentSelect.addEventListener("change", function () {
    currentStudentId = studentSelect.value;
    refreshLessons();
  });

  newStudentBtn.addEventListener("click", function () {
    newStudentForm.hidden = !newStudentForm.hidden;
  });

  newStudentForm.addEventListener("submit", async function (evt) {
    evt.preventDefault();
    const name = newStudentName.value.trim();
    if (!name) return;
    try {
      const resp = await fetch("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: name,
          grade_level: parseInt(newStudentGrade.value, 10) || 0,
          subjects: [subjectSelect.value],
          reading_level: "grade" + (parseInt(newStudentGrade.value, 10) || 0),
        }),
      });
      if (resp.ok) {
        newStudentName.value = "";
        newStudentGrade.value = "";
        newStudentForm.hidden = true;
        await refreshStudents();
      }
    } catch (err) {
      // non-fatal
    }
  });

  newLessonBtn.addEventListener("click", async function () {
    if (!currentStudentId) return;
    try {
      await fetch("/api/profiles/" + encodeURIComponent(currentStudentId) + "/lessons", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject: subjectSelect.value }),
      });
      await refreshLessons();
    } catch (err) {
      // non-fatal
    }
  });

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

  // -----------------------------------------------------------------------
  // 2026-09-20 attribution follow-up (docs/attribution_design.md): the host
  // attributes sentences to passages as a separate layer from the model's
  // own [S#] labels. It never edits or rewrites the answer text -- it only
  // adds small marker nodes (textContent/DOM nodes only, never raw markup) at sentence
  // boundaries: a subtle marker for a host-backed sentence (opens the
  // source viewer, same as a citation chip), a distinct style for an
  // unbacked sentence ("the tutor's own words -- not checked against the
  // library"), and a stronger one for unbacked_number (a figure not found
  // anywhere in the library). Spans are char offsets into the raw answer
  // text; renderTextWithCitations below never runs any markdown/HTML
  // transform on that text, so offsets always line up with what is on
  // screen -- a marker whose offset cannot be mapped onto the current text
  // (out of range) is simply skipped rather than mis-highlighting.
  // -----------------------------------------------------------------------

  function attributionMarkers(attributionsEvent) {
    const markers = [];
    if (!attributionsEvent) return markers;
    (attributionsEvent.attributions || []).forEach(function (a) {
      if (a.model_cited) return; // the model's own [S#] chip already covers this
      const span = a.sentence_span || [];
      markers.push({ pos: span[1], kind: "backed", passageId: a.passage_id });
    });
    (attributionsEvent.unbacked || []).forEach(function (u) {
      const span = u.span || [];
      markers.push({
        pos: span[1],
        kind: u.reason === "unbacked_number" ? "unbacked-number" : "unbacked",
      });
    });
    markers.sort(function (a, b) {
      return a.pos - b.pos;
    });
    return markers;
  }

  function makeAttributionMarkerNode(marker) {
    if (marker.kind === "backed") {
      const btn = el("button", {
        className: "attribution-marker attribution-backed",
        text: "●",
        attrs: { type: "button", "aria-label": "source-backed", "data-passage-id": marker.passageId || "" },
      });
      btn.addEventListener("click", function () {
        openSourceViewer(marker.passageId);
      });
      return btn;
    }
    if (marker.kind === "unbacked-number") {
      return el("span", {
        className: "attribution-marker attribution-unbacked-number",
        text: "⚠",
        attrs: { "aria-label": "number not found in the library" },
      });
    }
    return el("span", {
      className: "attribution-marker attribution-unbacked",
      text: "○",
      attrs: { "aria-label": "the tutor's own words -- not checked against the library" },
    });
  }

  // Rebuilds tutorNode's content from the final answer text, splitting on
  // both the model's own [S#] citation groups and the host's attribution
  // markers, in offset order. Only called once the answer is complete (on
  // the "attributions" event) so `text` is stable and matches the spans.
  function renderAnswerWithAttribution(container, text, attributionsEvent) {
    container.textContent = "";
    const markers = attributionMarkers(attributionsEvent).filter(function (m) {
      return typeof m.pos === "number" && m.pos >= 0 && m.pos <= text.length;
    });

    const citationMatches = [];
    CITATION_RE.lastIndex = 0;
    let match;
    while ((match = CITATION_RE.exec(text)) !== null) {
      citationMatches.push({
        start: match.index,
        end: CITATION_RE.lastIndex,
        labels: match[1].split(",").map(function (s) { return s.trim(); }),
      });
    }

    let cursor = 0;
    let markerIndex = 0;

    function emitTextUpTo(pos) {
      if (pos > cursor) {
        container.appendChild(document.createTextNode(text.slice(cursor, pos)));
        cursor = pos;
      }
    }

    function emitMarkersUpTo(pos) {
      while (markerIndex < markers.length && markers[markerIndex].pos <= pos) {
        emitTextUpTo(markers[markerIndex].pos);
        container.appendChild(makeAttributionMarkerNode(markers[markerIndex]));
        markerIndex += 1;
      }
    }

    citationMatches.forEach(function (citation) {
      emitMarkersUpTo(citation.start);
      emitTextUpTo(citation.start);
      citation.labels.forEach(function (label) {
        container.appendChild(renderCitationChip(label));
      });
      cursor = citation.end;
    });
    emitMarkersUpTo(text.length);
    emitTextUpTo(text.length);
  }

  // -----------------------------------------------------------------------
  // 2026-09-20 evidence-dump follow-up: "resolves" != "supports". The host
  // never edits or removes anything from the model's own text -- it only
  // collapses a detected dump behind a toggle, and adds a plain note when
  // every citation on the turn is unsupported. textContent/DOM nodes only.
  // -----------------------------------------------------------------------

  function collapseEvidenceDump(tutorNode) {
    const existingChildren = Array.prototype.slice.call(tutorNode.childNodes);
    const contentWrap = el("div", { className: "dump-content" });
    contentWrap.hidden = true;
    existingChildren.forEach(function (child) {
      contentWrap.appendChild(child);
    });
    const toggle = el("button", {
      className: "dump-toggle",
      text: "Show the tutor's pasted sources",
      attrs: { type: "button" },
    });
    toggle.addEventListener("click", function () {
      contentWrap.hidden = !contentWrap.hidden;
      toggle.textContent = contentWrap.hidden
        ? "Show the tutor's pasted sources"
        : "Hide the tutor's pasted sources";
    });
    tutorNode.appendChild(toggle);
    tutorNode.appendChild(contentWrap);
  }

  function appendUnsupportedSourcesNote() {
    const wrapper = el("div", { className: "msg msg-note" });
    const note = el("span", {
      className: "unsupported-note",
      text: "The tutor's sources did not match this question.",
    });
    wrapper.appendChild(note);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
  }

  // Bounded-generation follow-up (2026-09-20): the host may stop
  // generation early -- either the output token cap (`max_tokens`) or the
  // repetition-loop guard (tutor.app.repetition_guard) -- and marks the
  // turn `truncated` on the done event. The student only sees a short
  // plain note; the answer text itself is never edited beyond what the
  // host already trimmed. textContent/DOM nodes only, same as elsewhere.
  function appendTruncatedNote() {
    const wrapper = el("div", { className: "msg msg-note" });
    const note = el("span", {
      className: "truncated-note",
      text: "The tutor's answer was cut short.",
    });
    wrapper.appendChild(note);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
  }

  function applyCitationQuality(tutorNode, doneData, citationsEvent) {
    if (doneData && doneData.evidence_dump) {
      collapseEvidenceDump(tutorNode);
    }
    const citations = (citationsEvent && citationsEvent.citations) || [];
    const unsupportedLabels = (citationsEvent && citationsEvent.unsupported_labels) || [];
    if (citations.length > 0 && unsupportedLabels.length === citations.length) {
      appendUnsupportedSourcesNote();
    }
    if (doneData && doneData.truncated) {
      appendTruncatedNote();
    }
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

  function appendEvictionNote(data) {
    const wrapper = el("div", { className: "msg msg-note" });
    const note = el("span", {
      className: "eviction-note",
      text:
        "Trimmed " +
        (data.evicted_turns != null ? data.evicted_turns : "some") +
        " older turn(s) from context to make room (" +
        (data.tokens_before != null ? data.tokens_before : "?") +
        " -> " +
        (data.tokens_after != null ? data.tokens_after : "?") +
        " tokens).",
    });
    wrapper.appendChild(note);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
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
    lastCitationsEvent = null;
    lastAttributionsEvent = null;

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
      lastCitationsEvent = data;
      renderCitations(tutorNode, data.citations);
    } else if (eventName === "attributions") {
      lastAttributionsEvent = data;
      renderAnswerWithAttribution(tutorNode, getTutorLine(), data);
      renderCitations(tutorNode, lastCitationsEvent && lastCitationsEvent.citations);
    } else if (eventName === "eviction") {
      appendEvictionNote(data);
    } else if (eventName === "error") {
      appendError(data.message || "An error occurred.");
    } else if (eventName === "done") {
      lastTurnMeta = data;
      applyCitationQuality(tutorNode, data, lastCitationsEvent);
      refreshStatus();
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
    if (body.resources && body.resources.rss_mb != null) {
      rows.push(["RSS (MB)", String(Math.round(body.resources.rss_mb))]);
    }
    if (body.last_turn) {
      if (body.last_turn.route) rows.push(["Last route", String(body.last_turn.route)]);
      if (body.last_turn.cached_tokens != null) {
        rows.push(["Cached tokens", String(body.last_turn.cached_tokens)]);
      }
    } else if (lastTurnMeta && lastTurnMeta.route) {
      rows.push(["Last route", String(lastTurnMeta.route)]);
    }
    if (body.last_eviction) {
      rows.push([
        "Last eviction",
        (body.last_eviction.evicted_turns != null ? body.last_eviction.evicted_turns : "?") +
          " turn(s), " +
          (body.last_eviction.tokens_before != null ? body.last_eviction.tokens_before : "?") +
          " -> " +
          (body.last_eviction.tokens_after != null ? body.last_eviction.tokens_after : "?"),
      ]);
    }
    rows.forEach(function (pair) {
      statusPanelBody.appendChild(el("dt", { text: pair[0] }));
      statusPanelBody.appendChild(el("dd", { text: pair[1] }));
    });
  }

  ensureSession().then(refreshStatus);
  refreshStudents();
  setInterval(refreshStatus, 15000);
})();
