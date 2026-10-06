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

(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var site = window.FlySite;

  // Colours come from the theme tokens, so canvases follow light/dark.
  var C = {};
  function readColours() {
    var cs = getComputedStyle(document.documentElement);
    ["--ink", "--ink-2", "--ink-3", "--rule", "--rule-strong", "--accent", "--warn", "--heat",
     "--viz-bg", "--viz-hi", "--surface", "--bg"].forEach(function (k) { C[k.slice(2)] = cs.getPropertyValue(k).trim(); });
  }
  readColours();
  window.addEventListener("themechange", function () { setTimeout(readColours, 0); });
  site.colours = C;

  function rgb(hex) {
    var h = hex.replace("#", ""); if (h.length === 3) h = h.replace(/./g, "$&$&");
    var n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255];
  }
  site.rgb = rgb;

  // Canvas sized to its box at device resolution; returns [ctx, w, h].
  function fit(c) {
    var dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    var g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); return [g, w, h];
  }
  site.fit = fit;

  // ---------------------------------------------------------- buttons
  var LABELS = { "none": "Nothing", "right": "Right", "run-jump": "Right + jump", "run": "Right + run",
                 "run-jump-b": "Running jump", "jump": "Jump", "left": "Left" };
  var built = false;
  function buildButtons(actions) {
    $("buttons").innerHTML = actions.map(function (a, i) {
      return '<li id="b-' + i + '"><span>' + (LABELS[a] || a) + '</span><span class="bar"><i></i></span><span class="pct">0%</span></li>';
    }).join("");
    built = true;
  }

  // ------------------------------------------------------- attention
  var heatBuf = document.createElement("canvas");
  function drawHeat(att, grid) {
    var c = $("heat"), r = fit(c), g = r[0], w = r[1], h = r[2];
    g.clearRect(0, 0, w, h);
    if (!$("show-heat").checked || !att) return;
    var gx = grid[0], gy = grid[1];
    heatBuf.width = gx; heatBuf.height = gy;
    var hg = heatBuf.getContext("2d"), img = hg.createImageData(gx, gy), col = rgb(C.heat);
    for (var i = 0; i < att.length; i++) {
      var v = Math.max(0, (att[i] - 0.25) / 0.75);
      img.data[i * 4] = col[0]; img.data[i * 4 + 1] = col[1]; img.data[i * 4 + 2] = col[2];
      img.data[i * 4 + 3] = Math.round(200 * v * v);
    }
    hg.putImageData(img, 0, 0);
    // The fly's view covers screen rows 48-224 (below the score, above the floor's edge).
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(heatBuf, 0, h * 48 / 240, w, h * 176 / 240);
  }

  // ----------------------------------------------------------- frames
  var img = $("game"), loading = false;
  function pump() {
    if (loading) return;
    loading = true;
    var n = new Image();
    n.onload = function () { img.src = n.src; loading = false; setTimeout(pump, 40); };
    n.onerror = function () { loading = false; setTimeout(pump, 800); };
    n.src = "/api/frame/" + site.view() + ".jpg?t=" + Date.now();
  }

  // ------------------------------------------------------------ state
  var listeners = [];
  site.onState = function (fn) { listeners.push(fn); };
  function fmt(n) { return n == null ? "—" : Number(n).toLocaleString(); }

  function render(s) {
    var badge = $("badge");
    if (s.error) { badge.textContent = "The fly’s channel hit an error; it will restart."; return; }
    if (!s.ready) { badge.textContent = "Loading the brain…"; return; }
    if (s.idle || !s.level) {
      badge.textContent = site.view() === "theater" ? "No finished levels to show yet" : "Waiting for training";
      return;
    }
    badge.textContent = (site.view() === "live" ? "Learning " : "Replaying ") + "level " + s.level +
                        " · attempt " + s.attempt;
    if (!built && s.actions) buildButtons(s.actions);
    (s.probs || []).forEach(function (p, i) {
      var li = $("b-" + i); if (!li) return;
      li.className = s.actions[i] === s.action ? "on" : "";
      li.querySelector("i").style.width = (p * 100).toFixed(1) + "%";
      li.querySelector(".pct").textContent = Math.round(p * 100) + "%";
    });
    $("k-x").textContent = fmt(s.x);
    $("k-value").textContent = s.value == null ? "—" : (s.value >= 0 ? "+" : "") + s.value.toFixed(1);
    $("k-active").textContent = fmt(s.active);
    $("k-ckpt").textContent = s.ckpt_frames ? s.ckpt_frames.toFixed(1) + "M frames of practice" : "—";
    $("where").textContent = "Level " + s.level + " · distance " + fmt(s.x) + " · furthest " + fmt(s.max_x);
    $("flag").hidden = !(s.last_result && s.last_result.flag && s.x < 120 && s.last_result.level === s.level);
    drawHeat(s.attention, s.grid || [20, 16]);
    listeners.forEach(function (fn) { try { fn(s); } catch (e) { console.error(e); } });
  }

  var polling = false;
  function poll() {
    if (polling) return;
    polling = true;
    fetch("/api/state/" + site.view(), { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (s) { polling = false; render(s); setTimeout(poll, 120); })
      .catch(function () { polling = false; $("badge").textContent = "Reconnecting…"; setTimeout(poll, 1500); });
  }
  $("show-heat").addEventListener("change", function () { drawHeat(null); });
  window.addEventListener("viewchange", function () { built = false; $("game").removeAttribute("src"); });

  // --------------------------------------------------------- campaign
  var campListeners = [];
  site.onCampaign = function (fn) { campListeners.push(fn); };
  function pollCampaign() {
    fetch("/api/campaign", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (c) {
      $("f-level").textContent = c.current || "—";
      $("f-mastered").textContent = (c.mastered || []).length + " of " + (c.order || []).length;
      var t = c.training || [];
      $("f-frames").textContent = t.length ? t[t.length - 1].frames.toFixed(1) + "M frames" : "—";
      campListeners.forEach(function (fn) { try { fn(c); } catch (e) { console.error(e); } });
      setTimeout(pollCampaign, 5000);
    }).catch(function () { setTimeout(pollCampaign, 8000); });
  }

  pump(); poll(); pollCampaign();
})();
