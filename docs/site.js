// thunc site: syntax colouring, copy buttons and heading anchors. The pages read fine without it.
(function () {
  "use strict";

  var PY_KW = new Set(("and as assert async await break class continue def del elif else except finally for from global " +
    "if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
  var PY_TY = new Set("int str bool float list dict tuple set bytes None True False Literal Any Run".split(" "));
  var PY = /(#[^\n]*)|((?:[rRbBfFuU]{1,2})?(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(@[\w.]+)|(\b\d[\d_]*(?:\.\d+)?\b)|([A-Za-z_]\w*)/g;
  var SH = /((?:^|\s)#[^\n]*)|("(?:\\.|[^"\\])*"|'[^']*')|(^|\n|&& |\| )([A-Za-z_][\w.\/-]*)(?==)|(^|\n|&& |\| )([\w.\/-]+)/g;

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function span(cls, s) { return '<span class="tk-' + cls + '">' + esc(s) + "</span>"; }

  function python(src) {
    var out = "", last = 0, m;
    PY.lastIndex = 0;
    while ((m = PY.exec(src))) {
      out += esc(src.slice(last, m.index));
      if (m[1]) out += span("com", m[1]);
      else if (m[2]) out += span("str", m[2]);
      else if (m[3]) out += span("kw", m[3]);
      else if (m[4]) out += span("ty", m[4]);
      else {
        var w = m[5], next = src.charAt(PY.lastIndex);
        if (PY_KW.has(w)) out += span("kw", w);
        else if (PY_TY.has(w)) out += span("ty", w);
        else if (next === "(" && src.charAt(m.index - 1) !== "=") out += span("fn", w);
        else out += esc(w);
      }
      last = PY.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  function shell(src) {
    var out = "", last = 0, m;
    SH.lastIndex = 0;
    while ((m = SH.exec(src))) {
      out += esc(src.slice(last, m.index));
      if (m[1]) out += span("com", m[1]);
      else if (m[2]) out += span("str", m[2]);
      else if (m[4]) out += esc(m[3]) + span("ty", m[4]);
      else out += esc(m[5]) + span("fn", m[6]);
      last = SH.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  document.querySelectorAll("pre > code").forEach(function (code) {
    if (code.children.length) return; // already coloured by hand
    if (code.classList.contains("py")) code.innerHTML = python(code.textContent);
    else if (code.classList.contains("sh")) code.innerHTML = shell(code.textContent);
  });

  function copy(text, button) {
    try {
      navigator.clipboard.writeText(text).then(function () {
        var was = button.textContent;
        button.textContent = "Copied";
        setTimeout(function () { button.textContent = was; }, 1400);
      }, function () {});
    } catch (e) {}
  }

  document.querySelectorAll(".codeblock").forEach(function (block) {
    var pre = block.querySelector("pre");
    if (!pre) return;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.addEventListener("click", function () { copy(pre.textContent.replace(/[ \t]+$/gm, ""), button); });
    block.appendChild(button);
  });

  document.querySelectorAll("[data-copy]").forEach(function (button) {
    button.addEventListener("click", function () { copy(button.getAttribute("data-copy"), button); });
  });

  // On narrow screens the docs nav is a scrolling strip: bring the current page into view.
  var current = document.querySelector(".docs aside nav a.active");
  if (current && current.parentNode.scrollWidth > current.parentNode.clientWidth) {
    current.parentNode.scrollLeft = current.offsetLeft - current.parentNode.offsetLeft - 16;
  }

  document.querySelectorAll(".docs main h2[id], .docs main h3[id]").forEach(function (h) {
    var a = document.createElement("a");
    a.className = "anchor";
    a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to this section");
    a.textContent = "#";
    h.appendChild(a);
  });
})();
