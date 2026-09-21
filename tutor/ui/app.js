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
  const MARKDOWN_RE = /(\*\*([^*\n]+)\*\*)|(\*([^*\n]+)\*)|(`([^`\n]+)`)|(<br\s*\/?>)|(\n)/gi;

  // Inline-only tokenizer: bold/italic/code plus a literal "<br>" token
  // (some model answers put a literal <br> inside a table cell) both
  // rendered as a real <br> element. No block-level splitting here --
  // used for the text inside a single paragraph line, heading, list item
  // or table cell.
  function appendInlineMarkdown(container, text) {
    if (!text) return;
    let lastIndex = 0;
    let match;
    MARKDOWN_RE.lastIndex = 0;
    function plain(slice) {
      if (slice) container.appendChild(document.createTextNode(slice));
    }
    while ((match = MARKDOWN_RE.exec(text)) !== null) {
      plain(text.slice(lastIndex, match.index));
      if (match[1] !== undefined) {
        container.appendChild(el("strong", { text: match[2] }));
      } else if (match[3] !== undefined) {
        container.appendChild(el("em", { text: match[4] }));
      } else if (match[5] !== undefined) {
        container.appendChild(el("code", { text: match[6] }));
      } else if (match[7] !== undefined || match[8] !== undefined) {
        container.appendChild(el("br"));
      }
      lastIndex = MARKDOWN_RE.lastIndex;
    }
    plain(text.slice(lastIndex));
  }

  const _TABLE_ROW_RE = /^\s*\|.*\|\s*$/;
  const _TABLE_SEP_RE = /^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*$/;
  const _LIST_ITEM_RE = /^\s*([-*]|\d+\.)\s+(.*)$/;
  const _HEADING_RE = /^(#{1,6})\s+(.*)$/;

  function _splitTableRow(line) {
    let trimmed = line.trim();
    if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
    if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
    return trimmed.split("|").map(function (cell) {
      return cell.trim();
    });
  }

  function _buildTable(lines) {
    const wrapper = el("div", { className: "md-table-wrap" });
    const table = el("table", { className: "md-table" });
    const headCells = _splitTableRow(lines[0]);
    const thead = el("thead");
    const headRow = el("tr");
    headCells.forEach(function (cellText) {
      const th = el("th");
      appendInlineMarkdown(th, cellText);
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = el("tbody");
    for (let r = 2; r < lines.length; r++) {
      const cells = _splitTableRow(lines[r]);
      const tr = el("tr");
      cells.forEach(function (cellText) {
        const td = el("td");
        appendInlineMarkdown(td, cellText);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrapper.appendChild(table);
    return wrapper;
  }

  // Block-level markdown: pipe tables, `#`/`##`/... headings, `- `/`1. `
  // lists, and plain paragraphs, each rendered with createElement and
  // textContent DOM nodes only -- raw markup is never assigned. Inline
  // bold/italic/code (and a literal "<br>") still work inside every block
  // via appendInlineMarkdown above.
  function appendMarkdownText(container, text) {
    if (!text) return;
    const lines = text.split("\n");
    let i = 0;
    while (i < lines.length) {
      const headingMatch = _HEADING_RE.exec(lines[i]);
      if (headingMatch) {
        const tag = headingMatch[1].length <= 3 ? "h3" : "h4";
        const heading = el(tag, {});
        appendInlineMarkdown(heading, headingMatch[2]);
        container.appendChild(heading);
        i++;
        continue;
      }

      if (
        _TABLE_ROW_RE.test(lines[i]) &&
        i + 1 < lines.length &&
        _TABLE_SEP_RE.test(lines[i + 1])
      ) {
        const tableLines = [lines[i], lines[i + 1]];
        let j = i + 2;
        while (j < lines.length && _TABLE_ROW_RE.test(lines[j])) {
          tableLines.push(lines[j]);
          j++;
        }
        container.appendChild(_buildTable(tableLines));
        i = j;
        continue;
      }

      if (_LIST_ITEM_RE.test(lines[i])) {
        const ordered = /^\s*\d+\./.test(lines[i]);
        const list = el(ordered ? "ol" : "ul", {});
        let j = i;
        while (j < lines.length) {
          const itemMatch = _LIST_ITEM_RE.exec(lines[j]);
          if (!itemMatch) break;
          const li = el("li");
          appendInlineMarkdown(li, itemMatch[2]);
          list.appendChild(li);
          j++;
        }
        container.appendChild(list);
        i = j;
        continue;
      }

      // Paragraph: consume lines until the next block-starting line.
      const paraLines = [];
      let j = i;
      while (
        j < lines.length &&
        !_HEADING_RE.test(lines[j]) &&
        !_LIST_ITEM_RE.test(lines[j]) &&
        !(
          _TABLE_ROW_RE.test(lines[j]) &&
          j + 1 < lines.length &&
          _TABLE_SEP_RE.test(lines[j + 1])
        )
      ) {
        paraLines.push(lines[j]);
        j++;
      }
      if (j === i) {
        // Safety net: nothing matched a block and the paragraph loop
        // consumed zero lines (shouldn't happen given the checks above,
        // but never spin forever on unexpected input).
        appendInlineMarkdown(container, lines[i]);
        j = i + 1;
      } else {
        paraLines.forEach(function (line, idx) {
          appendInlineMarkdown(container, line);
          if (idx < paraLines.length - 1) container.appendChild(el("br"));
        });
      }
      i = j;
    }
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

  // -----------------------------------------------------------------------
  // 2026-09-20 offset-aware block mapping (tables/lists attribution
  // follow-up): `mapAnswerToBlocks` is a PURE function -- no DOM, no
  // globals besides the regexes already defined above -- that walks the
  // raw answer text the exact same way `appendMarkdownText` does (heading /
  // table / list-item / paragraph line grouping) but returns each block's
  // RAW character-offset range(s) into the original string instead of
  // building nodes. This lets `renderBlocksToDom` below place every
  // attribution marker and [S#] chip inside the correct cell/item/heading
  // by raw offset, instead of appending it as a sibling after a whole
  // table/list (which is what silently misplaced markers before this
  // change: appendMarkdownText built one opaque <table>/<ul> node per
  // block, so a marker appended afterward landed after the whole block,
  // never inside the row/item it belonged to).
  //
  // Block shapes returned:
  //   {type: "heading", tag, innerStart, innerEnd}
  //   {type: "table", headCells: [{start,end}], rows: [{start, end, cells: [{start,end}]}]}
  //   {type: "list", ordered, items: [{start, end, textStart, textEnd}]}
  //   {type: "paragraph", lines: [{start, end}]}
  // All offsets index the SAME raw text handed in -- never a slice.
  // -----------------------------------------------------------------------

  function _lineOffsets(text) {
    const lines = text.split("\n");
    const starts = [];
    let pos = 0;
    for (let idx = 0; idx < lines.length; idx++) {
      starts.push(pos);
      pos += lines[idx].length + 1;
    }
    return { lines: lines, starts: starts };
  }

  function _trimCellRange(text, s, e) {
    while (s < e && /\s/.test(text[s])) s++;
    while (e > s && /\s/.test(text[e - 1])) e--;
    return { start: s, end: e };
  }

  function _cellRanges(text, rowStart, rowEnd) {
    let s = rowStart;
    let e = rowEnd;
    if (text[s] === "|") s++;
    if (e > s && text[e - 1] === "|") e--;
    const cells = [];
    let cellStart = s;
    for (let k = s; k < e; k++) {
      if (text[k] === "|") {
        cells.push(_trimCellRange(text, cellStart, k));
        cellStart = k + 1;
      }
    }
    cells.push(_trimCellRange(text, cellStart, e));
    return cells;
  }

  function mapAnswerToBlocks(text) {
    const blocks = [];
    if (!text) return blocks;
    const lo = _lineOffsets(text);
    const lines = lo.lines;
    const starts = lo.starts;
    const n = lines.length;
    let i = 0;
    while (i < n) {
      const line = lines[i];

      const headingMatch = _HEADING_RE.exec(line);
      if (headingMatch) {
        const innerLen = headingMatch[2].length;
        const innerStart = starts[i] + (line.length - innerLen);
        blocks.push({
          type: "heading",
          tag: headingMatch[1].length <= 3 ? "h3" : "h4",
          innerStart: innerStart,
          innerEnd: innerStart + innerLen,
        });
        i++;
        continue;
      }

      if (
        _TABLE_ROW_RE.test(line) &&
        i + 1 < n &&
        _TABLE_SEP_RE.test(lines[i + 1])
      ) {
        const headStart = starts[i];
        const headEnd = headStart + line.length;
        const headCells = _cellRanges(text, headStart, headEnd);
        let j = i + 2;
        const rows = [];
        while (j < n && _TABLE_ROW_RE.test(lines[j])) {
          const rowStart = starts[j];
          const rowEnd = rowStart + lines[j].length;
          rows.push({ start: rowStart, end: rowEnd, cells: _cellRanges(text, rowStart, rowEnd) });
          j++;
        }
        blocks.push({ type: "table", headCells: headCells, rows: rows });
        i = j;
        continue;
      }

      if (_LIST_ITEM_RE.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const items = [];
        let j = i;
        while (j < n) {
          const itemMatch = _LIST_ITEM_RE.exec(lines[j]);
          if (!itemMatch) break;
          const lineStart = starts[j];
          const lineLen = lines[j].length;
          const textLen = itemMatch[2].length;
          const textStart = lineStart + (lineLen - textLen);
          items.push({
            start: lineStart,
            end: lineStart + lineLen,
            textStart: textStart,
            textEnd: textStart + textLen,
          });
          j++;
        }
        blocks.push({ type: "list", ordered: ordered, items: items });
        i = j;
        continue;
      }

      const paraLineIdx = [];
      let j = i;
      while (
        j < n &&
        !_HEADING_RE.test(lines[j]) &&
        !_LIST_ITEM_RE.test(lines[j]) &&
        !(
          _TABLE_ROW_RE.test(lines[j]) &&
          j + 1 < n &&
          _TABLE_SEP_RE.test(lines[j + 1])
        )
      ) {
        paraLineIdx.push(j);
        j++;
      }
      if (j === i) {
        paraLineIdx.push(i);
        j = i + 1;
      }
      blocks.push({
        type: "paragraph",
        lines: paraLineIdx.map(function (idx) {
          return { start: starts[idx], end: starts[idx] + lines[idx].length };
        }),
      });
      i = j;
    }
    return blocks;
  }

  // Renders the inline content of ONE block-piece (a heading's text, a
  // table cell, a list item's text, or one paragraph line) between raw
  // offsets [rangeStart, rangeEnd). `citationMatches` (each {start, end,
  // labels}) and `markers` (each {pos, ...}) are the FULL, absolute-offset
  // lists for the whole answer; only the ones whose offset falls inside
  // this range are consumed here, so a marker/chip belonging to another
  // block is never emitted at the wrong place. A marker's pos must be
  // `> rangeStart` (never at the very start of a range) and `<= rangeEnd`.
  function renderInlineSegment(container, text, rangeStart, rangeEnd, citationMatches, markers, resolveLabel, makeMarkerNode) {
    const events = [];
    (citationMatches || []).forEach(function (c) {
      if (c.start >= rangeStart && c.end <= rangeEnd) {
        events.push({ pos: c.start, order: 0, type: "citation", citation: c });
      }
    });
    (markers || []).forEach(function (m) {
      if (m.pos > rangeStart && m.pos <= rangeEnd) {
        events.push({ pos: m.pos, order: 1, type: "marker", marker: m });
      }
    });
    events.sort(function (a, b) {
      return a.pos - b.pos || a.order - b.order;
    });
    let cursor = rangeStart;
    events.forEach(function (ev) {
      if (ev.pos > cursor) {
        appendInlineMarkdown(container, text.slice(cursor, ev.pos));
        cursor = ev.pos;
      }
      if (ev.type === "citation") {
        ev.citation.labels.forEach(function (label) {
          container.appendChild(resolveLabel(label));
        });
        cursor = ev.citation.end;
      } else {
        container.appendChild(makeMarkerNode(ev.marker));
      }
    });
    if (cursor < rangeEnd) {
      appendInlineMarkdown(container, text.slice(cursor, rangeEnd));
    }
  }

  // Appends any marker whose pos falls within (start, end] to `container`
  // as-is (no text interleaving) -- used for a table row's last cell and a
  // list item, where the spec wants the marker at the END of the unit, not
  // interleaved mid-cell/mid-item.
  function appendMarkersInRange(container, markers, start, end, makeMarkerNode) {
    (markers || []).forEach(function (m) {
      if (m.pos > start && m.pos <= end) {
        container.appendChild(makeMarkerNode(m));
      }
    });
  }

  // Renders the full block list produced by `mapAnswerToBlocks` into
  // `container`, wiring markers/chips into the correct cell/item/heading/
  // paragraph line. A heading gets no markers (only its own [S#] chips, if
  // any -- an empty markers list is passed). A table row's marker(s) land
  // in its LAST cell; a list item's marker(s) land at the end of the item.
  function renderBlocksToDom(container, text, blocks, citationMatches, markers, resolveLabel, makeMarkerNode) {
    blocks.forEach(function (block) {
      if (block.type === "heading") {
        const heading = el(block.tag, {});
        renderInlineSegment(heading, text, block.innerStart, block.innerEnd, citationMatches, [], resolveLabel, makeMarkerNode);
        container.appendChild(heading);
        return;
      }
      if (block.type === "table") {
        const wrapper = el("div", { className: "md-table-wrap" });
        const table = el("table", { className: "md-table" });
        const thead = el("thead");
        const headRow = el("tr");
        block.headCells.forEach(function (cellRange) {
          const th = el("th");
          renderInlineSegment(th, text, cellRange.start, cellRange.end, citationMatches, [], resolveLabel, makeMarkerNode);
          headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);

        const tbody = el("tbody");
        block.rows.forEach(function (row) {
          const tr = el("tr");
          row.cells.forEach(function (cellRange, idx) {
            const td = el("td");
            renderInlineSegment(td, text, cellRange.start, cellRange.end, citationMatches, [], resolveLabel, makeMarkerNode);
            if (idx === row.cells.length - 1) {
              appendMarkersInRange(td, markers, row.start, row.end, makeMarkerNode);
            }
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrapper.appendChild(table);
        container.appendChild(wrapper);
        return;
      }
      if (block.type === "list") {
        const list = el(block.ordered ? "ol" : "ul", {});
        block.items.forEach(function (item) {
          const li = el("li");
          renderInlineSegment(li, text, item.textStart, item.textEnd, citationMatches, [], resolveLabel, makeMarkerNode);
          appendMarkersInRange(li, markers, item.start, item.end, makeMarkerNode);
          list.appendChild(li);
        });
        container.appendChild(list);
        return;
      }
      // paragraph
      block.lines.forEach(function (lineRange, idx) {
        renderInlineSegment(container, text, lineRange.start, lineRange.end, citationMatches, markers, resolveLabel, makeMarkerNode);
        if (idx < block.lines.length - 1) container.appendChild(el("br"));
      });
    });
  }

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

    function resolveLabel(label) {
      if (unresolvedLabels[label]) return renderUnresolvedLabel(label);
      return renderCitationChip(label, passageIdByLabel[label]);
    }

    // Block/cell/item-aware placement (2026-09-20 tables/lists attribution
    // follow-up): map the raw text into blocks with real offset ranges
    // first, then hand each block-piece only the markers/chips whose raw
    // offset actually falls inside it. A marker whose offset cannot be
    // mapped onto any block-piece (should not happen given the filter
    // above, but never trust it blindly) is simply never emitted, rather
    // than misplaced.
    const blocks = mapAnswerToBlocks(text);
    renderBlocksToDom(container, text, blocks, citationMatches, markers, resolveLabel, makeAttributionMarkerNode);
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

  // ---------------------------------------------------------------------
  // Working (busy) bubble: shown immediately on send, replaced by the
  // streamed answer at first token, or by a plain retry-able error on
  // failure. See docs feedback "NO FEEDBACK AFTER SEND" (2026-09-20).
  // ---------------------------------------------------------------------

  const STATUS_TEXT = {
    searching: "Looking in the library...",
    reading: "Reading sources...",
    thinking: "Writing an answer...",
    tool: "Working...",
    checking: "Checking the answer against the sources...",
  };

  function appendWorkingBubble() {
    const wrapper = el("div", { className: "msg msg-tutor msg-working" });
    const dots = el("span", { className: "working-dots", attrs: { "aria-hidden": "true" } });
    dots.appendChild(el("span", { className: "dot" }));
    dots.appendChild(el("span", { className: "dot" }));
    dots.appendChild(el("span", { className: "dot" }));
    wrapper.appendChild(dots);
    const statusLine = el("span", {
      className: "working-status",
      text: "Looking in the library...",
      attrs: { "aria-live": "polite" },
    });
    wrapper.appendChild(statusLine);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;

    const startedAt = Date.now();
    const timer = setInterval(function () {
      const elapsedMs = Date.now() - startedAt;
      if (elapsedMs < 3000) return;
      const seconds = Math.floor(elapsedMs / 1000);
      const base = statusLine.getAttribute("data-base") || statusLine.textContent;
      statusLine.setAttribute("data-base", base);
      statusLine.textContent = base + " ... " + seconds + " s";
    }, 1000);

    return {
      node: wrapper,
      statusLine: statusLine,
      setStatus: function (stage, detail) {
        const text = detail || STATUS_TEXT[stage] || "Working...";
        statusLine.setAttribute("data-base", text);
        statusLine.textContent = text;
      },
      stop: function () {
        clearInterval(timer);
      },
    };
  }

  function replaceWorkingBubbleWithError(working, message) {
    working.stop();
    working.node.className = "msg msg-error msg-retry";
    working.node.textContent = "";
    working.node.appendChild(
      el("span", { text: message || "Something went wrong. " })
    );
    const retryBtn = el("button", { className: "retry-btn", text: "Retry" });
    retryBtn.type = "button";
    working.node.appendChild(retryBtn);
    return retryBtn;
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
    let tutorNode = null;
    let firstTokenSeen = false;
    const working = appendWorkingBubble();
    lastCitationsEvent = null;
    lastAttributionsEvent = null;

    function ensureTutorNode() {
      if (!firstTokenSeen) {
        firstTokenSeen = true;
        working.stop();
        working.node.remove();
        tutorNode = appendMessage("tutor", "");
      }
      return tutorNode;
    }

    currentAbortController = new AbortController();
    try {
      const resp = await fetch("/api/session/" + sid + "/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: currentAbortController.signal,
      });

      if (!resp.ok) {
        const retryBtn = replaceWorkingBubbleWithError(
          working,
          "Request failed (" + resp.status + ")."
        );
        retryBtn.addEventListener("click", function () {
          working.node.remove();
          sendTurn(payload);
        });
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
          handleFrame(
            frame,
            function () {
              return ensureTutorNode();
            },
            function (text) {
              tutorLine = text;
            },
            function () {
              return tutorLine;
            },
            working,
            function (message) {
              if (firstTokenSeen) {
                appendError(message);
                return;
              }
              const retryBtn = replaceWorkingBubbleWithError(working, message);
              retryBtn.addEventListener("click", function () {
                working.node.remove();
                sendTurn(payload);
              });
            }
          );
        }
      }
    } catch (err) {
      if (!firstTokenSeen) {
        const retryBtn = replaceWorkingBubbleWithError(working, "Connection lost.");
        retryBtn.addEventListener("click", function () {
          working.node.remove();
          sendTurn(payload);
        });
      } else {
        appendError("Connection lost.");
      }
    } finally {
      working.stop();
      setBusy(false);
      currentAbortController = null;
    }
  }

  function handleFrame(frame, getTutorNode, setTutorLine, getTutorLine, working, onError) {
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

    if (eventName === "status") {
      // Additive: unknown to older clients/eval harnesses, safe to ignore
      // there. Here it drives the working bubble's live status line until
      // the first token arrives.
      if (working) working.setStatus(data.stage, data.detail);
      return;
    }

    const tutorNode = getTutorNode();

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
      if (onError) {
        onError(data.message || "Something went wrong.");
      } else {
        appendError(data.message || "An error occurred.");
      }
    } else if (eventName === "done") {
      lastTurnMeta = data;
      applyCitationQuality(tutorNode, data, lastCitationsEvent, lastAttributionsEvent);
      refreshStatus();
    }
  }

  function setBusy(isBusy) {
    sendBtn.disabled = isBusy;
    sendBtn.classList.toggle("is-busy", isBusy);
    input.disabled = isBusy;
    stopBtn.disabled = !isBusy;
    actionButtons.forEach(function (btn) {
      btn.disabled = isBusy;
    });
  }

  composer.addEventListener("submit", function (evt) {
    evt.preventDefault();
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;
    appendMessage("user", text);
    input.value = "";
    sendTurn({ text: text });
  });

  input.addEventListener("keydown", function (evt) {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      if (typeof composer.requestSubmit === "function") {
        composer.requestSubmit();
      } else {
        composer.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    }
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

  // "Show more of the article" state for the source panel currently open.
  // Reset on every openSourceViewer call; before/after are cumulative char
  // counts sent to the /context endpoint, growing by _CONTEXT_PAGE_CHARS
  // each time "Show earlier"/"Show later" is used.
  const _CONTEXT_PAGE_CHARS = 1500;
  let currentContextState = null;

  async function openSourceViewer(passageId, answerSentence) {
    sourceViewerBody.textContent = "";
    sourceViewerBody.appendChild(el("p", { text: "Loading..." }));
    currentContextState = null;
    try {
      const resp = await fetch("/api/source/" + encodeURIComponent(passageId));
      if (!resp.ok) {
        sourceViewerBody.textContent = "";
        sourceViewerBody.appendChild(el("p", { text: "Source not found." }));
        return;
      }
      const body = await resp.json();
      renderSourceView(body, answerSentence, passageId);
    } catch (err) {
      sourceViewerBody.textContent = "";
      sourceViewerBody.appendChild(el("p", { text: "Could not load source." }));
    }
  }

  async function fetchArticleContext(passageId, before, after) {
    const resp = await fetch(
      "/api/source/" +
        encodeURIComponent(passageId) +
        "/context?before=" +
        encodeURIComponent(before) +
        "&after=" +
        encodeURIComponent(after)
    );
    if (resp.status === 503) {
      const body = await resp.json().catch(function () {
        return {};
      });
      return { unavailable: true, message: body.message || "The book is not available right now." };
    }
    if (!resp.ok) {
      return { unavailable: true, message: "Could not load more of the article." };
    }
    return await resp.json();
  }

  function renderExpandedContext(container, data, answerSentence) {
    container.textContent = "";
    if (data.unavailable) {
      container.appendChild(el("p", { className: "source-context-error", text: data.message }));
      return;
    }
    const text = data.text || "";
    const passage = data.passage || { start: 0, end: 0 };
    const sentenceSpan = findBestSentenceSpan(text, passage.start, passage.end, answerSentence);
    const start = sentenceSpan ? sentenceSpan.start : passage.start;
    const end = sentenceSpan ? sentenceSpan.end : passage.end;

    const p = el("p", { className: "source-context-text" });
    p.appendChild(document.createTextNode(text.slice(0, start)));
    const highlight = el("span", {
      className: "source-highlight",
      attrs: { title: "What the tutor saw" },
      text: text.slice(start, end),
    });
    p.appendChild(highlight);
    p.appendChild(document.createTextNode(text.slice(end)));
    container.appendChild(p);

    const label = el("p", { className: "source-context-label", text: "What the tutor saw is highlighted above." });
    container.appendChild(label);

    const pagingRow = el("div", { className: "source-context-paging" });
    if (data.more_before) {
      const earlierBtn = el("button", { className: "source-context-page", text: "Show earlier" });
      earlierBtn.type = "button";
      earlierBtn.addEventListener("click", function () {
        expandContext(answerSentence, { before: true });
      });
      pagingRow.appendChild(earlierBtn);
    }
    if (data.more_after) {
      const laterBtn = el("button", { className: "source-context-page", text: "Show later" });
      laterBtn.type = "button";
      laterBtn.addEventListener("click", function () {
        expandContext(answerSentence, { after: true });
      });
      pagingRow.appendChild(laterBtn);
    }
    if (pagingRow.childNodes.length) {
      container.appendChild(pagingRow);
    }
  }

  async function expandContext(answerSentence, grow) {
    if (!currentContextState) return;
    if (grow && grow.before) {
      currentContextState.before += _CONTEXT_PAGE_CHARS;
    }
    if (grow && grow.after) {
      currentContextState.after += _CONTEXT_PAGE_CHARS;
    }
    const data = await fetchArticleContext(
      currentContextState.passageId,
      currentContextState.before,
      currentContextState.after
    );
    renderExpandedContext(currentContextState.container, data, answerSentence);
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

  function renderSourceView(body, answerSentence, passageId) {
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

    // "Show more of the article" (owner-requested reader affordance): the
    // stored passage is a 400-char word-packed chunk that starts/ends
    // mid-sentence; this expands it to surrounding article text fetched
    // fresh from /api/source/{id}/context, keeping the model-received
    // passage highlighted inside the longer text.
    const expandContainer = el("div", { className: "source-context" });
    expandContainer.style.display = "none";
    const toggleBtn = el("button", {
      className: "source-context-toggle",
      text: "Show more of the article ▾",
    });
    toggleBtn.type = "button";
    toggleBtn.setAttribute("aria-expanded", "false");
    toggleBtn.addEventListener("click", async function () {
      const expanded = toggleBtn.getAttribute("aria-expanded") === "true";
      if (expanded) {
        expandContainer.style.display = "none";
        expandContainer.textContent = "";
        toggleBtn.setAttribute("aria-expanded", "false");
        toggleBtn.textContent = "Show more of the article ▾";
        currentContextState = null;
        return;
      }
      toggleBtn.setAttribute("aria-expanded", "true");
      toggleBtn.textContent = "▴ Show less";
      expandContainer.style.display = "";
      expandContainer.textContent = "";
      expandContainer.appendChild(el("p", { text: "Loading more of the article..." }));
      currentContextState = {
        passageId: passageId,
        container: expandContainer,
        before: _CONTEXT_PAGE_CHARS,
        after: _CONTEXT_PAGE_CHARS,
      };
      const data = await fetchArticleContext(passageId, _CONTEXT_PAGE_CHARS, _CONTEXT_PAGE_CHARS);
      renderExpandedContext(expandContainer, data, answerSentence);
    });
    sourceViewerBody.appendChild(toggleBtn);
    sourceViewerBody.appendChild(expandContainer);
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

  // Exposes the pure offset-mapping function (no DOM access) to a Node.js
  // test harness, when one is loading this file as a CommonJS module. A
  // real browser never defines `module`, so this is a no-op there.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { mapAnswerToBlocks: mapAnswerToBlocks };
  }
})();
