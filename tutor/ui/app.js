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

  // -----------------------------------------------------------------------
  // 2026-09-20 safe minimal markdown (docs bug #4): **bold**, *italic*,
  // `code`, line breaks/paragraphs, and simple "- "/"1. " list lines.
  // Built with createElement/textContent DOM nodes only -- never raw markup
  // -- and driven off the SAME raw string that attribution/citation
  // offsets index into, so callers always hand this a raw slice and the
  // offsets used elsewhere in this file (markers, citation spans) never
  // have to account for markdown syntax being stripped: they are computed
  // against the untouched raw text before any slice reaches here.
  // -----------------------------------------------------------------------
  const MARKDOWN_RE = /(\*\*([^*\n]+)\*\*)|(\*([^*\n]+)\*)|(`([^`\n]+)`)|(\n)/g;

  function appendMarkdownText(container, text) {
    if (!text) return;
    let lastIndex = 0;
    let match;
    let currentLine = container;
    MARKDOWN_RE.lastIndex = 0;
    function plain(slice) {
      if (slice) currentLine.appendChild(document.createTextNode(slice));
    }
    while ((match = MARKDOWN_RE.exec(text)) !== null) {
      plain(text.slice(lastIndex, match.index));
      if (match[1] !== undefined) {
        currentLine.appendChild(el("strong", { text: match[2] }));
      } else if (match[3] !== undefined) {
        currentLine.appendChild(el("em", { text: match[4] }));
      } else if (match[5] !== undefined) {
        currentLine.appendChild(el("code", { text: match[6] }));
      } else if (match[7] !== undefined) {
        currentLine.appendChild(el("br"));
      }
      lastIndex = MARKDOWN_RE.lastIndex;
    }
    plain(text.slice(lastIndex));
  }

  function renderTextWithCitations(container, text) {
    let lastIndex = 0;
    let match;
    CITATION_RE.lastIndex = 0;
    while ((match = CITATION_RE.exec(text)) !== null) {
      if (match.index > lastIndex) {
        appendMarkdownText(container, text.slice(lastIndex, match.index));
      }
      const labels = match[1].split(",").map(function (s) { return s.trim(); });
      labels.forEach(function (label) {
        container.appendChild(renderCitationChip(label));
      });
      lastIndex = CITATION_RE.lastIndex;
    }
    if (lastIndex < text.length) {
      appendMarkdownText(container, text.slice(lastIndex));
    }
  }

  function renderCitations(container, citations) {
    // A label the model wrote that matches no evidence passage (e.g.
    // [S89] when only S1-S5 were ever handed to it) is `unresolved`;
    // render it as plain muted text, not a clickable source-viewer chip,
    // and never count it alongside real citations (see applyCitationQuality).
    (citations || []).forEach(function (citation) {
      const label = citation.label || citation;
      if (citation.unresolved) {
        container.appendChild(renderUnresolvedLabel(label));
        return;
      }
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

  // Renders a computed number for display: plain for an integer, rounded
  // to 4 decimal places (trailing zeros dropped by String()) otherwise --
  // never full float noise like 79.99999999999999.
  function formatComputedNumber(n) {
    if (typeof n !== "number" || !isFinite(n)) return String(n);
    if (Number.isInteger(n)) return String(n);
    return String(Math.round(n * 10000) / 10000);
  }

  function attributionMarkers(attributionsEvent, text) {
    const markers = [];
    if (!attributionsEvent) return markers;
    (attributionsEvent.attributions || []).forEach(function (a) {
      if (a.model_cited) return; // the model's own [S#] chip already covers this
      const span = a.sentence_span || [];
      markers.push({
        pos: span[1],
        kind: "backed",
        passageId: a.passage_id,
        sentenceText: typeof text === "string" ? text.slice(span[0], span[1]) : "",
      });
    });
    (attributionsEvent.unbacked || []).forEach(function (u) {
      const span = u.span || [];
      markers.push({
        pos: span[1],
        kind: u.reason === "unbacked_number" ? "unbacked-number" : "unbacked",
      });
    });
    // 2026-09-20 computed-statement follow-up (docs/calc_investigation.md
    // fix #1): the host verifies stated arithmetic against the calc
    // evaluator. Only items with a span land inline as markers here --
    // a "the question asked for a number the answer never gave" item
    // (span: null) is rendered as a separate note, see appendComputedGapNote.
    (attributionsEvent.computed || []).forEach(function (c) {
      if (!c.span) return;
      markers.push({
        pos: c.span[1],
        kind: c.status === "verified" ? "computed-verified" : "computed-mismatch",
        computed: c.computed,
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
        attrs: {
          type: "button",
          "aria-label": "Found in the sources the tutor looked up",
          title: "Found in the sources the tutor looked up",
          "data-passage-id": marker.passageId || "",
        },
      });
      btn.addEventListener("click", function () {
        openSourceViewer(marker.passageId, marker.sentenceText);
      });
      return btn;
    }
    if (marker.kind === "unbacked-number") {
      return el("span", {
        className: "attribution-marker attribution-unbacked-number",
        text: "⚠",
        attrs: {
          "aria-label": "This number is not in the sources the tutor looked up — double-check it",
          title: "This number is not in the sources the tutor looked up — double-check it",
        },
      });
    }
    if (marker.kind === "computed-verified") {
      return el("span", {
        className: "attribution-marker attribution-computed-verified",
        text: "✓ checked",
        attrs: { "aria-label": "Checked by the calculator", title: "Checked by the calculator" },
      });
    }
    if (marker.kind === "computed-mismatch") {
      return el("span", {
        className: "attribution-marker attribution-computed-mismatch",
        text: "The calculator gets " + formatComputedNumber(marker.computed),
        attrs: { "aria-label": "the calculator disagrees with this number" },
      });
    }
    return el("span", {
      className: "attribution-marker attribution-unbacked",
      text: "○",
      attrs: {
        "aria-label": "Not found in the sources the tutor looked up — this may be the tutor's own knowledge",
        title: "Not found in the sources the tutor looked up — this may be the tutor's own knowledge",
      },
    });
  }

  // A computed-check mismatch the host detected in the STUDENT'S QUESTION
  // (the answer never stated the number the question asked for at all,
  // span: null) -- rendered as its own note rather than an inline marker
  // since there is no answer text position to attach it to. textContent
  // only, like every other note in this file.
  function appendComputedGapNote(computed) {
    const wrapper = el("div", { className: "msg msg-note" });
    const note = el("span", {
      className: "attribution-marker attribution-computed-mismatch",
      text: "The calculator gets " + formatComputedNumber(computed),
    });
    wrapper.appendChild(note);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
  }

  // Rebuilds tutorNode's content from the final answer text, splitting on
  // both the model's own [S#] citation groups and the host's attribution
  // markers, in offset order. Only called once the answer is complete (on
  // the "attributions" event) so `text` is stable and matches the spans.
  function renderAnswerWithAttribution(container, text, attributionsEvent, citationsEvent) {
    container.textContent = "";
    const markers = attributionMarkers(attributionsEvent, text).filter(function (m) {
      return typeof m.pos === "number" && m.pos >= 0 && m.pos <= text.length;
    });
    // Single source of truth for a label's resolution: the `citations`
    // event. Every render path (streaming, citations-only, full
    // attribution rebuild) looks the label up here so there is exactly
    // one chip per label occurrence in the text, and a resolved chip
    // always carries the real passage_id (never falls back to the bare
    // label, which is what produced the "Source not found" duplicate).
    const unresolvedLabels = {};
    const passageIdByLabel = {};
    ((citationsEvent && citationsEvent.citations) || []).forEach(function (c) {
      if (c.unresolved) {
        unresolvedLabels[c.label] = true;
      } else {
        passageIdByLabel[c.label] = c.passage_id;
      }
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
        appendMarkdownText(container, text.slice(cursor, pos));
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
        if (unresolvedLabels[label]) {
          container.appendChild(renderUnresolvedLabel(label));
        } else {
          container.appendChild(renderCitationChip(label, passageIdByLabel[label]));
        }
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

  // 2026-09-20 wording follow-up: two different situations, two different
  // notes. `attributionsEvent.passages_available` (additive field, see
  // compose.py) is 0 when research/retrieval returned nothing at all for
  // this question -- the library genuinely had nothing to check against.
  // When passages WERE available but none of them backed anything the
  // model said, that is a different (narrower, more accurate) claim: only
  // THIS answer went unchecked, not that the library is empty.
  function appendUnsupportedSourcesNote(attributionsEvent) {
    const noPassagesAtAll =
      attributionsEvent && attributionsEvent.passages_available === 0;
    const wrapper = el("div", { className: "msg msg-note" });
    const note = el("span", {
      className: "unsupported-note",
      text: noPassagesAtAll
        ? "The tutor did not find anything in the library for this question."
        : "None of this answer was found in the sources the tutor looked up.",
    });
    wrapper.appendChild(note);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
  }

  // Is there at least one sentence the HOST verified against a passage or
  // the calculator on this turn? This is a UI/UX question, separate from
  // compose.py's `citation_quality` (a model-behaviour metric the soak/eval
  // code depends on and which this file never redefines). A sentence with
  // a model-written [S#] that also matches an evidence passage counts as
  // model_cited=True in the attributions event; a host-attributed sentence
  // with no model label counts too. Either way, if the host actually
  // backed something, the "did not match" note is simply wrong and must
  // not show, even when the model mislabeled which sentence gets the chip.
  function hasHostBackedContent(attributionsEvent) {
    if (!attributionsEvent) return false;
    const backedByModel = (attributionsEvent.attributions || []).some(function (a) {
      return a.model_cited;
    });
    const backedByHost = (attributionsEvent.attributions || []).some(function (a) {
      return !a.model_cited && a.passage_id;
    });
    const verifiedComputed = (attributionsEvent.computed || []).some(function (c) {
      return c.status === "verified";
    });
    return backedByModel || backedByHost || verifiedComputed;
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

  function applyCitationQuality(tutorNode, doneData, citationsEvent, attributionsEvent) {
    if (doneData && doneData.evidence_dump) {
      collapseEvidenceDump(tutorNode);
    }
    // 2026-09-20 attribution-driven note (see docs/attribution_design.md):
    // compose.py's `citation_quality` ("uncited"/"unsupported"/"ok") is a
    // model-behaviour metric the soak/eval code depends on and is left
    // untouched server-side. The UI note is a different question -- "did
    // the HOST actually check anything against the library on this turn?"
    // -- answered from the `attributions` event, not from citation_quality,
    // because the model can staple a citation label onto the wrong
    // sentence while the host still genuinely backs another sentence to
    // the same (or another) passage. Only when nothing was backed at all,
    // and nothing was verified by the calculator, do we tell the student
    // this answer came from the model's own knowledge.
    if (!hasHostBackedContent(attributionsEvent)) {
      appendUnsupportedSourcesNote(attributionsEvent);
    }
    if (doneData && doneData.truncated) {
      appendTruncatedNote();
    }
  }

  function renderUnresolvedLabel(label) {
    return el("span", {
      className: "citation-label-unresolved",
      text: "[" + label + "]",
      attrs: { "aria-label": "citation label not found in the evidence for this turn" },
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
      // Rebuild (not append) from the raw text: appending here alongside
      // the inline chips already drawn during token streaming, or
      // alongside the rebuild below on "attributions", is exactly what
      // produced two chips for one [S1] occurrence.
      renderAnswerWithAttribution(tutorNode, getTutorLine(), lastAttributionsEvent, data);
    } else if (eventName === "attributions") {
      lastAttributionsEvent = data;
      renderAnswerWithAttribution(tutorNode, getTutorLine(), data, lastCitationsEvent);
      (data.computed || []).forEach(function (c) {
        if (!c.span && c.status === "mismatch") {
          appendComputedGapNote(c.computed);
        }
      });
    } else if (eventName === "eviction") {
      appendEvictionNote(data);
    } else if (eventName === "error") {
      appendError(data.message || "An error occurred.");
    } else if (eventName === "done") {
      lastTurnMeta = data;
      applyCitationQuality(tutorNode, data, lastCitationsEvent, lastAttributionsEvent);
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

  async function openSourceViewer(passageId, answerSentence) {
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
      renderSourceView(body, answerSentence);
    } catch (err) {
      sourceViewerBody.textContent = "";
      sourceViewerBody.appendChild(el("p", { text: "Could not load source." }));
    }
  }

  // 2026-09-20 sentence-level highlight (bug #5): the server highlight
  // covers the whole passage the retriever matched; clicking a per-sentence
  // ● marker should narrow that down to the sentence sharing the most
  // terms with the clicked answer sentence, computed here from text the
  // server already sent (no new endpoint). Falls back to the server's
  // whole-passage highlight whenever there's no answer sentence to compare
  // against, no passage text to search, or no sentence scores above zero.
  function wordsOf(s) {
    return (String(s || "").toLowerCase().match(/[a-z0-9]+/g)) || [];
  }

  function findBestSentenceSpan(text, searchStart, searchEnd, answerSentence) {
    if (!text || !answerSentence) return null;
    const region = text.slice(searchStart, searchEnd);
    const sentenceRe = /[^.!?]+[.!?]*/g;
    const answerWords = new Set(wordsOf(answerSentence));
    if (!answerWords.size) return null;
    let m;
    let best = null;
    let bestScore = 0;
    while ((m = sentenceRe.exec(region)) !== null) {
      if (!m[0].trim()) continue;
      const words = wordsOf(m[0]);
      let score = 0;
      words.forEach(function (w) {
        if (answerWords.has(w)) score += 1;
      });
      if (score > bestScore) {
        bestScore = score;
        best = { start: searchStart + m.index, end: searchStart + m.index + m[0].length };
      }
    }
    return bestScore > 0 ? best : null;
  }

  function renderSourceView(body, answerSentence) {
    sourceViewerBody.textContent = "";

    const heading = el("h3", { text: body.title || "" });
    sourceViewerBody.appendChild(heading);

    if (body.heading_path && body.heading_path.length) {
      sourceViewerBody.appendChild(el("p", { className: "source-path", text: body.heading_path.join(" > ") }));
    }

    const fullText = body.text || "";
    const wholeStart = body.highlight ? body.highlight.start : 0;
    const wholeEnd = body.highlight ? body.highlight.end : fullText.length;
    const sentenceSpan = findBestSentenceSpan(fullText, wholeStart, wholeEnd, answerSentence);
    const start = sentenceSpan ? sentenceSpan.start : wholeStart;
    const end = sentenceSpan ? sentenceSpan.end : wholeEnd;

    const passage = el("p");
    const contextBefore = fullText
      ? fullText.slice(Math.max(0, start - 200), start)
      : body.context_before || "";
    const contextAfter = fullText
      ? fullText.slice(end, end + 200)
      : body.context_after || "";
    const highlightText = fullText ? fullText.slice(start, end) : "";

    passage.appendChild(document.createTextNode(contextBefore));

    const highlightSpan = el("span", {
      className: "source-highlight",
      text: highlightText,
    });
    passage.appendChild(highlightSpan);

    passage.appendChild(document.createTextNode(contextAfter));
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
