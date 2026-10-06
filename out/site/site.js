(function () {
  "use strict";

  // ---------------------------------------------------------------- theme
  var root = document.documentElement;
  function storedTheme() { try { return localStorage.getItem("theme") || "system"; } catch (e) { return "system"; } }
  function setTheme(t) {
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t); else root.removeAttribute("data-theme");
    try { localStorage.setItem("theme", t); } catch (e) {}
    document.querySelectorAll("[data-theme-set]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.themeSet === t));
    });
    window.dispatchEvent(new Event("themechange"));
  }
  document.querySelectorAll("[data-theme-set]").forEach(function (b) {
    b.addEventListener("click", function () { setTheme(b.dataset.themeSet); });
  });
  setTheme(storedTheme());
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(function () {
      window.dispatchEvent(new Event("themechange"));
    });
  }

  // ----------------------------------------------------------------- view
  var view = "live";
  function setView(v) {
    view = v;
    document.querySelectorAll(".tabs [role=tab]").forEach(function (b) {
      b.setAttribute("aria-selected", String(b.dataset.view === v));
    });
    document.querySelectorAll(".view-theater").forEach(function (el) { el.hidden = v !== "theater"; });
    try { localStorage.setItem("view", v); } catch (e) {}
    window.dispatchEvent(new Event("viewchange"));
  }
  document.querySelectorAll(".tabs [role=tab]").forEach(function (b) {
    b.addEventListener("click", function () { setView(b.dataset.view); });
  });
  try { var sv = localStorage.getItem("view"); if (sv === "theater") setView("theater"); } catch (e) {}

  window.FlySite = { view: function () { return view; }, setView: setView };
})();
