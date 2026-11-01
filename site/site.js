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
  try {
    var qv = (location.search.match(/[?&]view=(live|theater)/) || [])[1];
    var sv = qv || localStorage.getItem("view"); if (sv === "theater") setView("theater");
  } catch (e) {}

  // Replay mode: the page is static (GitHub Pages) and plays recorded runs.
  var replay = !!document.querySelector('meta[name="smartfly-mode"][content="replay"]');
  if (replay) root.classList.add("replay");

  window.FlySite = { view: function () { return view; }, setView: setView, replay: replay };
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
    badge.textContent = s.badge || (site.view() === "live" ? "Learning " : "Replaying ") + "level " +
                        s.level + " · attempt " + s.attempt;
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
    $("flag").hidden = !(s.show_flag ||
      (s.last_result && s.last_result.flag && s.x < 120 && s.last_result.level === s.level));
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
      showCampaign(c);
      setTimeout(pollCampaign, 5000);
    }).catch(function () { setTimeout(pollCampaign, 8000); });
  }
  function showCampaign(c) {
    $("f-level").textContent = c.current || "—";
    $("f-mastered").textContent = (c.mastered || []).length + " of " + (c.order || []).length;
    var t = c.training || [];
    $("f-frames").textContent = t.length ? t[t.length - 1].frames.toFixed(1) + "M frames" : "—";
    campListeners.forEach(function (fn) { try { fn(c); } catch (e) { console.error(e); } });
  }

  // ----------------------------------------------------------- replay
  // Plays recorded runs (replays/<id>/run.json and one JPEG per decision) through
  // the same render path the live site uses, at the same fifteen decisions a second.
  var CONDITION = { normal: "", blind: "blind", scrambled: "scrambled eyes" };
  var index = null, runs = {}, current = null, timer = null;
  site.pick = { level: null, condition: "normal" };

  function bytes(b64) {
    var s = atob(b64), out = new Array(s.length);
    for (var i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) / 255;
    return out;
  }
  function chosenRun() {
    var all = index.runs;
    if (site.view() === "live") return all.filter(function (r) { return r.practice; })[0] || all[0];
    var lv = site.pick.level || (all.filter(function (r) { return !r.practice; })[0] || {}).level;
    return all.filter(function (r) {
      return !r.practice && r.level === lv && r.condition === site.pick.condition;
    })[0] || all[0];
  }
  function load(meta, done) {
    if (runs[meta.id]) return done(runs[meta.id]);
    fetch("replays/" + meta.id + "/run.json").then(function (r) { return r.json(); }).then(function (run) {
      run.frames = run.steps.map(function (_, i) {
        var im = new Image(); im.src = "replays/" + meta.id + "/" + String(i).padStart(4, "0") + ".jpg"; return im;
      });
      runs[meta.id] = run; done(run);
    });
  }
  function describe(run) {
    var what = run.practice ? "Practising level " + run.level : "Level " + run.level +
      (CONDITION[run.condition] ? " · " + CONDITION[run.condition] : "");
    return what + " · recorded " + new Date(run.recorded * 1000).toLocaleDateString([], { day: "numeric", month: "short" });
  }
  function play() {
    clearTimeout(timer);
    if (!index || !index.runs.length) { $("badge").textContent = "No recordings yet"; return; }
    var meta = chosenRun(); current = meta.id;
    $("badge").textContent = "Loading the recording…";
    load(meta, function (run) {
      if (current !== meta.id) return;
      run.practice = meta.practice;
      var i = 0, trace = [], hold = 0;
      var history = index.runs.filter(function (r) { return !r.practice; }).map(function (r) {
        return { t: r.recorded, level: r.level, x: r.x, flag: r.flag, note: CONDITION[r.condition] };
      });
      (function tick() {
        if (current !== meta.id) return;
        var st = run.steps[i], im = run.frames[i];
        if (!im.complete) { timer = setTimeout(tick, 30); return; }  // wait for the frame
        if (hold === 0) {
          trace.push([st.r, st.v, st.ev]); if (trace.length > 160) trace.shift();
          img.src = im.src;
          render({
            ready: true, level: run.level, attempt: run.attempts, badge: describe(run),
            x: st.x, max_x: st.max_x, actions: run.actions, action: run.actions[st.a], probs: st.p,
            value: st.v, "return": st.ret, reward_trace: trace.slice(), ckpt_frames: run.ckpt_frames,
            eye: bytes(st.eye), eye_shape: run.eye_shape, photo: bytes(st.photo), deep: bytes(st.deep),
            attention: bytes(st.att), grid: run.grid, raster: st.ras, raster_n: run.raster_n,
            active: st.act, show_flag: st.ev === "flag", history: history, queue: []
          });
        }
        if (i < run.steps.length - 1) { i++; timer = setTimeout(tick, 1000 / 15); return; }
        // Hold the last moment, then play it again from the start.
        if (++hold < 30) { timer = setTimeout(tick, 1000 / 15); return; }
        i = 0; hold = 0; trace = []; timer = setTimeout(tick, 400);
      })();
    });
  }
  site.replayAgain = play;

  if (site.replay) {
    fetch("replays/index.json").then(function (r) { return r.json(); }).then(function (data) {
      index = data; showCampaign(data.campaign); play();
    }).catch(function () { $("badge").textContent = "Couldn’t load the recordings."; });
    window.addEventListener("viewchange", play);
  } else {
    pump(); poll(); pollCampaign();
  }
})();

(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var site = window.FlySite, C = site.colours, fit = site.fit, rgb = site.rgb;

  // Draw a grid of values as a small image scaled up crisply. Each map keeps a
  // slowly moving range so its brightness does not flicker frame to frame.
  var ranges = {};
  var buf = document.createElement("canvas");
  function gridMap(id, vals, cols, rows, tint) {
    var c = $(id); if (!c || !vals) return;
    var r = fit(c), g = r[0], w = r[1], h = r[2];
    // Robust range (5th to 98th percentile): a handful of saturated neurons
    // would otherwise stretch the scale and leave the rest black.
    var sorted = Array.prototype.slice.call(vals).sort(function (x, y) { return x - y; });
    var lo = sorted[Math.floor(sorted.length * 0.05)], hi = sorted[Math.floor(sorted.length * 0.98)];
    var rg = ranges[id] || (ranges[id] = { lo: lo, hi: hi });
    rg.lo += (lo - rg.lo) * 0.2; rg.hi += (hi - rg.hi) * 0.2;
    var span = Math.max(rg.hi - rg.lo, 1e-6);
    var a = rgb(C["viz-bg"]), b = rgb(tint || C["viz-hi"]);
    buf.width = cols; buf.height = rows;
    var bg = buf.getContext("2d"), img = bg.createImageData(cols, rows);
    for (var k = 0; k < cols * rows; k++) {
      var t = Math.max(0, Math.min(1, ((vals[k] || 0) - rg.lo) / span));
      t = Math.pow(t, 0.8);
      for (var ch = 0; ch < 3; ch++) img.data[k * 4 + ch] = Math.round(a[ch] + (b[ch] - a[ch]) * t);
      img.data[k * 4 + 3] = 255;
    }
    bg.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(buf, 0, 0, w, h);
  }

  // Scrolling record of which sampled neurons were strongly active.
  var cols = [], COLS = 160;
  function raster(ids, n) {
    var c = $("v-raster"), r = fit(c), g = r[0], w = r[1], h = r[2];
    cols.push(ids || []); if (cols.length > COLS) cols.shift();
    g.fillStyle = C["viz-bg"]; g.fillRect(0, 0, w, h);
    var cw = w / COLS, rh = h / Math.max(n, 1);
    g.fillStyle = C.ink;
    for (var j = 0; j < cols.length; j++) {
      var x = w - (cols.length - j) * cw, col = cols[j];
      g.globalAlpha = 0.35 + 0.65 * (j / cols.length);
      for (var k = 0; k < col.length; k++) g.fillRect(x, col[k] * rh, Math.max(cw - 0.5, 1), Math.max(rh, 1));
    }
    g.globalAlpha = 1;
  }

  // Reward per decision as bars around zero, with the fly's own expectation as a line.
  function rewards(trace, total) {
    var c = $("v-reward"), r = fit(c), g = r[0], w = r[1], h = r[2];
    g.clearRect(0, 0, w, h);
    var L = 34, R = 6, T = 14, B = 8, W = w - L - R, H = h - T - B, N = 160;
    var hi = 4; (trace || []).forEach(function (e) { hi = Math.max(hi, e[0], e[1]); });
    hi = Math.ceil(hi / 2) * 2; var lo = -2;
    function Y(v) { return T + H * (hi - Math.max(lo, Math.min(hi, v))) / (hi - lo); }
    g.font = "11px " + getComputedStyle(document.body).getPropertyValue("--mono");
    g.textAlign = "right"; g.fillStyle = C["ink-3"]; g.strokeStyle = C.rule; g.lineWidth = 1;
    [lo, 0, hi / 2, hi].forEach(function (v) {
      g.beginPath(); g.moveTo(L, Math.round(Y(v)) + .5); g.lineTo(L + W, Math.round(Y(v)) + .5); g.stroke();
      g.fillText(v > 0 ? "+" + v : String(v), L - 7, Y(v) + 4);
    });
    if (!trace || !trace.length) return;
    var bw = W / N, x0 = L + W - trace.length * bw, y0 = Y(0);
    trace.forEach(function (e, i) {
      var y = Y(e[0]), x = x0 + i * bw;
      g.fillStyle = e[0] >= 0 ? C.accent : C.warn;
      g.fillRect(x, Math.min(y, y0), Math.max(bw - 1, 1), Math.max(Math.abs(y - y0), 1));
      if (e[2]) {
        g.fillStyle = e[2] === "flag" ? C.accent : C.warn; g.textAlign = "center";
        g.fillText(e[2] === "flag" ? "flag" : "died", x + bw / 2, 10);
      }
    });
    g.strokeStyle = C.heat; g.lineWidth = 1.6; g.beginPath();
    trace.forEach(function (e, i) { var x = x0 + (i + .5) * bw, y = Y(e[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
    $("ret").textContent = "This attempt: " + (total >= 0 ? "+" : "") + total.toFixed(1);
  }

  var lastAttempt = null;
  site.onState(function (s) {
    if (s.attempt !== lastAttempt) { cols = []; lastAttempt = s.attempt; }
    var es = s.eye_shape || [15, 24];
    gridMap("v-eye", s.eye, es[1], es[0]);
    gridMap("v-photo", s.photo || s.eye, es[1], es[0]);
    gridMap("v-deep", s.deep, 20, 16);
    gridMap("v-attn", s.attention, 20, 16, C.heat);
    raster(s.raster, s.raster_n || 480);
    rewards(s.reward_trace, s["return"] || 0);
  });
  window.addEventListener("viewchange", function () { cols = []; ranges = {}; });
})();

(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var site = window.FlySite, C = site.colours, fit = site.fit;
  var camp = null, playing = null;
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function clock(t) { return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

  // --------------------------------------------------------- world map
  function worlds(c) {
    var html = "";
    for (var w = 1; w <= 8; w++) {
      html += '<div class="world"><h4>World ' + w + '</h4><ol>';
      for (var l = 1; l <= 4; l++) {
        var lv = w + "-" + l, k = "";
        if ((c.mastered || []).indexOf(lv) >= 0) k = "mastered";
        else if ((c.parked || []).indexOf(lv) >= 0) k = "parked";
        if (lv === c.current) k = "current";
        var title = k === "mastered" ? "Mastered" : k === "current" ? "Learning now" : k === "parked" ? "Set aside for now" : "Not reached yet";
        html += '<li class="' + k + '" title="' + lv + ": " + title + '">' + lv + "</li>";
      }
      html += "</ol></div>";
    }
    $("worlds").innerHTML = html;
  }

  function tests(c) {
    var t = (c.tests || []).slice().reverse();
    $("tests").innerHTML = t.length ? t.map(function (x) {
      var pass = x.flags >= 8;
      return '<li><time>' + clock(x.t) + '</time><span class="lv">' + esc(x.level) + '</span><span>' +
        x.flags + " of " + x.of + " runs reached the flag · average distance " + Number(x.mean_x).toLocaleString() +
        '</span><span class="' + (pass ? "ok" : "no") + '">' + (pass ? "Mastered" : "Not yet") + "</span></li>";
    }).join("") : '<li><span></span><span></span><span class="no">No tests yet.</span><span></span></li>';
  }

  // ------------------------------------------------------------ charts
  function chart(id, rows, key, max, fmt, ref) {
    var c = $(id), r = fit(c), g = r[0], w = r[1], h = r[2];
    g.clearRect(0, 0, w, h);
    var L = 44, R = 8, T = 8, B = 22, W = w - L - R, H = h - T - B;
    g.font = "11px IBM Plex Mono, ui-monospace, monospace"; g.textAlign = "right";
    g.strokeStyle = C.rule; g.fillStyle = C["ink-3"]; g.lineWidth = 1;
    for (var k = 0; k <= 4; k++) {
      var y = Math.round(T + H * (1 - k / 4)) + .5;
      g.beginPath(); g.moveTo(L, y); g.lineTo(L + W, y); g.stroke(); g.fillText(fmt(max * k / 4), L - 8, y + 4);
    }
    if (!rows.length) return;
    var x0 = rows[0].frames, x1 = Math.max(rows[rows.length - 1].frames, x0 + 0.1);
    g.textAlign = "left"; g.fillText(x0.toFixed(1) + "M", L, h - 5);
    g.textAlign = "right"; g.fillText(x1.toFixed(1) + "M frames", L + W, h - 5);
    if (ref) {
      var ry = T + H * (1 - ref / max); g.strokeStyle = C["rule-strong"]; g.setLineDash([3, 4]);
      g.beginPath(); g.moveTo(L, ry); g.lineTo(L + W, ry); g.stroke(); g.setLineDash([]);
    }
    g.strokeStyle = C.accent; g.lineWidth = 1.8; g.beginPath();
    rows.forEach(function (row, i) {
      var px = L + W * (row.frames - x0) / (x1 - x0), py = T + H * (1 - Math.min(key(row), max) / max);
      if (i) g.lineTo(px, py); else g.moveTo(px, py);
    });
    g.stroke();
  }

  // ----------------------------------------------------------- theater
  function picker(c) {
    var m = c.mastered || [], all = c.order || [];
    $("picker").innerHTML = all.map(function (lv) {
      var ok = m.indexOf(lv) >= 0;
      return '<button data-level="' + lv + '"' + (ok ? "" : " disabled") + (lv === playing ? ' class="playing"' : "") +
        ' title="' + (ok ? "Watch the fly play " + lv : lv + " is not mastered yet") + '">' + lv + "</button>";
    }).join("");
  }
  $("picker").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-level]"); if (!b || b.disabled) return;
    var lv = b.dataset.level;
    if (site.replay) {
      site.pick.level = lv; playing = lv; if (camp) picker(camp);
      if (site.view() !== "theater") site.setView("theater"); else site.replayAgain();
      return;
    }
    $("queue-note").textContent = "Asking for " + lv + "…";
    fetch("/api/theater", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ level: lv }) })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (!res.ok) { $("queue-note").textContent = res.reason || "That didn’t work."; return; }
        $("queue-note").textContent = res.position === 0 ? lv + " is playing now." :
          lv + " is in the queue, position " + res.position + ".";
        if (site.view() !== "theater") site.setView("theater");
      })
      .catch(function () { $("queue-note").textContent = "Couldn’t reach the fly. Try again in a moment."; });
  });

  document.querySelectorAll("#conditions button").forEach(function (b) {
    b.addEventListener("click", function () {
      site.pick.condition = b.dataset.condition;
      document.querySelectorAll("#conditions button").forEach(function (o) {
        o.setAttribute("aria-pressed", String(o === b));
      });
      site.replayAgain();
    });
  });

  site.onState(function (s) {
    if (site.view() !== "theater") return;
    if (s.level !== playing) { playing = s.level; if (camp) picker(camp); }
    var q = s.queue || [];
    if (!site.replay && q.length && !/position|playing now|Asking/.test($("queue-note").textContent)) {
      $("queue-note").textContent = "Up next: " + q.join(", ");
    }
    $("runs").innerHTML = (s.history || []).slice(0, 12).map(function (r) {
      return '<li><time>' + clock(r.t) + '</time><span class="lv">' + esc(r.level) + '</span><span>distance ' +
        Number(r.x).toLocaleString() + (r.note ? " · " + esc(r.note) : "") +
        '</span><span class="' + (r.flag ? "ok" : "no") + '">' +
        (r.flag ? "Reached the flag" : "Didn’t finish") + "</span></li>";
    }).join("") || '<li><span></span><span></span><span class="no">Runs will appear here.</span><span></span></li>';
  });

  function redraw() {
    if (!camp) return;
    var rows = camp.training || [];
    chart("c-rate", rows, function (r) { return 100 * r.flags / Math.max(r.of, 1); }, 100,
          function (v) { return Math.round(v) + "%"; }, 80);
    var mx = 1000; rows.forEach(function (r) { mx = Math.max(mx, r.mean_x); });
    mx = Math.ceil(mx / 500) * 500;
    chart("c-dist", rows, function (r) { return r.mean_x; }, mx, function (v) { return Math.round(v).toLocaleString(); });
  }
  site.onCampaign(function (c) { camp = c; worlds(c); tests(c); picker(c); redraw(); });
  window.addEventListener("themechange", function () { setTimeout(redraw, 0); });
  window.addEventListener("resize", redraw);
})();
