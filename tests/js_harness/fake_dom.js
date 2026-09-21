// Minimal fake DOM used to execute tutor/ui/app.js's pure/DOM-lite logic
// under a pure JS engine (no Node, no browser). Deliberately small: just
// enough surface for app.js to load at top level and for the exported
// functions to build/inspect node trees. Throws on any innerHTML
// assignment so a regression that starts using innerHTML fails loudly.

function makeNode(tagName) {
  const node = {
    tagName: tagName || null,
    nodeType: tagName ? 1 : 3,
    _text: "",
    className: "",
    children: [],
    attrs: {},
    style: {},
    listeners: {},
    hidden: false,
    disabled: false,
    value: "",
    type: "",
  };
  Object.defineProperty(node, "textContent", {
    get: function () {
      if (node.nodeType === 3) return node._text;
      return node.children.map(function (c) { return c.textContent; }).join("");
    },
    set: function (v) {
      node._text = v;
      node.children = [];
      if (v) {
        const t = makeNode(null);
        t._text = v;
        node.children.push(t);
      }
    },
  });
  Object.defineProperty(node, "innerHTML", {
    get: function () { return ""; },
    set: function () {
      throw new Error("innerHTML must never be assigned");
    },
  });
  node.appendChild = function (child) {
    if (child.parentNode && child.parentNode.children) {
      const oldIdx = child.parentNode.children.indexOf(child);
      if (oldIdx >= 0) child.parentNode.children.splice(oldIdx, 1);
    }
    node.children.push(child);
    child.parentNode = node;
    return child;
  };
  Object.defineProperty(node, "childNodes", {
    get: function () { return node.children; },
  });
  node.removeChild = function (child) {
    const idx = node.children.indexOf(child);
    if (idx >= 0) node.children.splice(idx, 1);
  };
  node.remove = function () {};
  node.setAttribute = function (k, v) { node.attrs[k] = String(v); };
  node.getAttribute = function (k) {
    return Object.prototype.hasOwnProperty.call(node.attrs, k) ? node.attrs[k] : null;
  };
  node.addEventListener = function (evt, fn) {
    node.listeners[evt] = node.listeners[evt] || [];
    node.listeners[evt].push(fn);
  };
  node.removeEventListener = function () {};
  node.dispatchEvent = function (evt) {
    (node.listeners[evt.type] || []).forEach(function (fn) { fn(evt); });
    return true;
  };
  node.classList = {
    _set: {},
    add: function (c) { this._set[c] = true; },
    remove: function (c) { delete this._set[c]; },
    toggle: function (c, force) {
      const on = force !== undefined ? force : !this._set[c];
      if (on) this._set[c] = true; else delete this._set[c];
      return on;
    },
    contains: function (c) { return !!this._set[c]; },
  };
  node.querySelectorAll = function () { return []; };
  node.querySelector = function () { return null; };
  return node;
}

const _byId = {};
const document = {
  getElementById: function (id) {
    if (!_byId[id]) _byId[id] = makeNode("div");
    return _byId[id];
  },
  createElement: function (tag) { return makeNode(tag); },
  createTextNode: function (text) {
    const n = makeNode(null);
    n._text = text;
    return n;
  },
  addEventListener: function () {},
  querySelectorAll: function () { return []; },
  readyState: "complete",
};

const window = {};
function fetchStub() {
  return Promise.resolve({
    ok: false,
    status: 0,
    json: function () { return Promise.resolve({}); },
  });
}
function EventSourceStub() {}
function AbortControllerStub() { this.signal = {}; }
AbortControllerStub.prototype.abort = function () {};

function setIntervalStub() { return 0; }
function clearIntervalStub() {}
function TextDecoderStub() {}
TextDecoderStub.prototype.decode = function (v) { return v || ""; };

const fetch = fetchStub;
const EventSource = EventSourceStub;
const AbortController = AbortControllerStub;
const setInterval = setIntervalStub;
const clearInterval = clearIntervalStub;
const TextDecoder = TextDecoderStub;

function _mkMarker(cls) {
  var b = document.createElement("button");
  b.className = cls;
  return b;
}
function _mkSpan() {
  return document.createElement("span");
}

var module = { exports: {} };
