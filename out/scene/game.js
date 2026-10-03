/* The runner, ported from src/gym/subway.py.
 *
 * It has to be an exact port, not a lookalike: the brain's readout was fitted
 * against the retina this code produces, so any drift in lane geometry, obstacle
 * timing or the projective depth mapping would feed it a distribution it has
 * never seen. Constants here mirror the Python module one for one.
 */
(function (global) {
  "use strict";

  var EMPTY = 0, TRAIN = 1, LOW = 2, HIGH = 3, COIN = 4;
  var A_NONE = 0, A_LEFT = 1, A_RIGHT = 2, A_JUMP = 3, A_ROLL = 4;
  var N_LANES = 3, RETINA_H = 24, RETINA_W = 15;

  // Deterministic RNG so a seed reproduces a run exactly.
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
  }

  function Runner(seed) {
    this.depth = 24;
    this.jumpFrames = 6;
    this.rollFrames = 6;
    this.baseSpeed = 0.34;
    this.maxSpeed = 1.0;
    this.speedRamp = 4000;
    this.coinRate = 0.06;
    this.laneChangeFrames = 2;
    this.rand = rng(seed || 1);
    this.retina = new Float32Array(RETINA_H * RETINA_W);
    this.reset();
  }

  Runner.prototype.spawnRow = function () {
    var row = [0, 0, 0];
    var safe = (this.rand() * N_LANES) | 0;
    var difficulty = Math.min(1, this.distance / 3000);
    var fill = 0.25 + 0.45 * difficulty;
    for (var lane = 0; lane < N_LANES; lane++) {
      if (lane === safe) continue;
      if (this.rand() < fill) {
        var wTrain = 0.34 + 0.3 * difficulty, tot = wTrain + 0.33 + 0.33;
        var r = this.rand() * tot;
        row[lane] = r < wTrain ? TRAIN : (r < wTrain + 0.33 ? LOW : HIGH);
      }
    }
    if (this.rand() < this.coinRate) {
      var free = [];
      for (var l = 0; l < N_LANES; l++) if (row[l] === EMPTY) free.push(l);
      if (free.length) row[free[(this.rand() * free.length) | 0]] = COIN;
    }
    return row;
  };

  Runner.prototype.reset = function () {
    this.track = [];
    this.lane = 1; this.targetLane = 1; this.laneTimer = 0;
    this.air = 0; this.low = 0;
    this.distance = 0; this.scroll = 0; this.coins = 0;
    this.frames = 0; this.done = false;
    for (var d = 0; d < this.depth; d++) {
      this.track.push(d > 6 ? this.spawnRow() : [0, 0, 0]);
    }
    this.observe();
    return this.retina;
  };

  Object.defineProperty(Runner.prototype, "speed", {
    get: function () {
      var ramp = Math.min(1, this.distance / this.speedRamp);
      return this.baseSpeed + (this.maxSpeed - this.baseSpeed) * ramp;
    }
  });

  Runner.prototype.step = function (action) {
    if (this.done) return;
    this.frames++;

    if (action === A_LEFT && this.laneTimer === 0 && this.targetLane > 0) {
      this.targetLane--; this.laneTimer = this.laneChangeFrames;
    } else if (action === A_RIGHT && this.laneTimer === 0 && this.targetLane < N_LANES - 1) {
      this.targetLane++; this.laneTimer = this.laneChangeFrames;
    } else if (action === A_JUMP && this.air === 0 && this.low === 0) {
      this.air = this.jumpFrames;
    } else if (action === A_ROLL && this.low === 0 && this.air === 0) {
      this.low = this.rollFrames;
    }

    if (this.laneTimer > 0) {
      this.laneTimer--;
      if (this.laneTimer === 0) this.lane = this.targetLane;
    }
    if (this.air > 0) this.air--;
    if (this.low > 0) this.low--;

    this.scroll += this.speed;
    var advance = Math.floor(this.scroll);
    this.scroll -= advance;

    for (var a = 0; a < advance; a++) {
      var arriving = this.track[0];
      this.track.shift();
      this.track.push(this.spawnRow());
      this.distance += 1;

      var occupied = [this.lane];
      if (this.laneTimer > 0 && this.targetLane !== this.lane) occupied.push(this.targetLane);
      for (var oi = 0; oi < occupied.length; oi++) {
        var cell = arriving[occupied[oi]];
        if (cell === COIN) this.coins++;
        else if (cell === TRAIN) this.done = true;
        else if (cell === LOW && this.air === 0) this.done = true;
        else if (cell === HIGH && this.low === 0) this.done = true;
      }
      if (this.done) break;
    }
    this.observe();
  };

  /* Projective depth mapping, matching the Python renderer: near track occupies
   * many more rows than far track, so an imminent obstacle dominates the field. */
  Runner.prototype.observe = function () {
    var img = this.retina;
    img.fill(0);
    var band = (RETINA_W / N_LANES) | 0;
    var viewH = RETINA_H - 3;
    var k = 3.0, edge = [];
    for (var d = 0; d <= this.depth; d++) {
      edge.push(Math.round(viewH * (1 - k / (k + d))));
    }
    var intensity = [0, 1.0, 0.66, 0.40, 0.18];
    for (var dd = 0; dd < Math.min(this.depth, viewH); dd++) {
      var r1 = viewH - edge[dd], r0 = viewH - edge[dd + 1];
      if (r1 <= r0) r0 = Math.max(0, r1 - 1);
      for (var lane = 0; lane < N_LANES; lane++) {
        var cell = this.track[dd][lane];
        if (cell === EMPTY) continue;
        var v = intensity[cell];
        for (var y = Math.max(r0, 0); y < Math.max(r1, 0); y++) {
          for (var x = lane * band; x < lane * band + band; x++) img[y * RETINA_W + x] = v;
        }
      }
    }
    // Self-state rows: lane, airborne, rolling.
    var x0 = this.lane * band;
    for (var sx = x0; sx < x0 + band; sx++) img[viewH * RETINA_W + sx] = 1.0;
    if (this.laneTimer > 0) {
      var tx = this.targetLane * band;
      for (var tx2 = tx; tx2 < tx + band; tx2++) img[viewH * RETINA_W + tx2] = 0.5;
    }
    if (this.air > 0) for (var ax = 0; ax < RETINA_W; ax++)
      img[(viewH + 1) * RETINA_W + ax] = this.air / this.jumpFrames;
    if (this.low > 0) for (var lx = 0; lx < RETINA_W; lx++)
      img[(viewH + 2) * RETINA_W + lx] = this.low / this.rollFrames;
    return img;
  };

  /* ------------------------------------------------------------- rendering */
  /* Drawn to a canvas used as the phone's screen texture. Deliberately styled
   * like the real game -- three rails running to a vanishing point -- rather
   * than like the retina, which is what the fly sees, not what a player sees. */
  Runner.prototype.draw = function (ctx, W, H) {
    var g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#6fb7e8"); g.addColorStop(0.42, "#9bd0ee");
    g.addColorStop(0.43, "#5b4536"); g.addColorStop(1, "#3b2a20");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

    var horizon = H * 0.43, cx = W / 2;
    var nearHalf = W * 0.46, farHalf = W * 0.055;
    function proj(d) {                       // depth -> screen row and half-width
      var t = 1 - 3.0 / (3.0 + d);
      var y = horizon + (H - horizon) * (1 - t);
      var hw = farHalf + (nearHalf - farHalf) * (1 - t);
      return [y, hw];
    }
    function laneX(lane, hw) { return cx + (lane - 1) * (hw * 0.62); }

    // Rails.
    ctx.strokeStyle = "rgba(220,220,230,.55)"; ctx.lineWidth = Math.max(1, W * 0.006);
    [-1, 0, 1].forEach(function (off) {
      ctx.beginPath();
      for (var d = 0; d <= 22; d++) {
        var p = proj(d), x = cx + off * (p[1] * 0.62);
        if (d === 0) ctx.moveTo(x, p[0]); else ctx.lineTo(x, p[0]);
      }
      ctx.stroke();
    });
    // Sleepers.
    ctx.strokeStyle = "rgba(70,48,34,.75)";
    for (var d2 = 0; d2 < 22; d2 += 1) {
      var p2 = proj(d2);
      ctx.lineWidth = Math.max(1, (1 - d2 / 22) * W * 0.012);
      ctx.beginPath();
      ctx.moveTo(cx - p2[1] * 0.95, p2[0]); ctx.lineTo(cx + p2[1] * 0.95, p2[0]);
      ctx.stroke();
    }

    // Obstacles, far to near so nearer ones overlap correctly.
    var COL = {1: ["#c9ced6", "#8f959e"], 2: ["#d8603c", "#a33f22"], 3: ["#e0b23a", "#a97f17"]};
    for (var dd = Math.min(this.depth, 22) - 1; dd >= 0; dd--) {
      for (var lane = 0; lane < 3; lane++) {
        var cell = this.track[dd][lane];
        if (!cell) continue;
        var pn = proj(dd), pf = proj(dd + 1);
        var hw = pn[1], x = laneX(lane, hw), w = hw * 0.50;
        var yN = pn[0], yF = pf[0];
        if (cell === COIN) {
          ctx.fillStyle = "#ffd34d";
          ctx.beginPath(); ctx.arc(x, yN - hw * 0.22, Math.max(1.5, hw * 0.10), 0, 6.283);
          ctx.fill();
          continue;
        }
        var c = COL[cell];
        var h = cell === TRAIN ? hw * 1.25 : (cell === LOW ? hw * 0.34 : hw * 0.30);
        var top = cell === HIGH ? yF - hw * 1.15 : yN - h;
        ctx.fillStyle = c[0];
        ctx.fillRect(x - w / 2, top, w, h);
        ctx.fillStyle = c[1];
        ctx.fillRect(x - w / 2, top, w, Math.max(1, h * 0.16));
        if (cell === TRAIN) {
          ctx.fillStyle = "rgba(150,200,235,.85)";
          ctx.fillRect(x - w * 0.32, top + h * 0.20, w * 0.64, h * 0.20);
        }
      }
    }

    // Runner.
    var pr = proj(0.6), hwR = pr[1], px = laneX(this.lane, hwR);
    if (this.laneTimer > 0) {
      var frac = 1 - this.laneTimer / this.laneChangeFrames;
      px = laneX(this.lane, hwR) + (laneX(this.targetLane, hwR) - laneX(this.lane, hwR)) * frac;
    }
    var py = pr[0] - (this.air > 0 ? hwR * 0.55 * Math.sin(Math.PI * (1 - this.air / this.jumpFrames)) : 0);
    var ph = this.low > 0 ? hwR * 0.26 : hwR * 0.48;
    ctx.fillStyle = "#f0d7b8";
    ctx.fillRect(px - hwR * 0.12, py - ph, hwR * 0.24, ph);
    ctx.fillStyle = "#2f6fd0";
    ctx.fillRect(px - hwR * 0.13, py - ph * 0.52, hwR * 0.26, ph * 0.40);
    ctx.fillStyle = "#e2453a";
    ctx.beginPath(); ctx.arc(px, py - ph - hwR * 0.04, hwR * 0.11, 0, 6.283); ctx.fill();

    // HUD.
    ctx.fillStyle = "rgba(255,255,255,.95)";
    ctx.font = "600 " + Math.round(H * 0.030) + "px ui-monospace,monospace";
    ctx.textAlign = "left";
    ctx.fillText(String(Math.floor(this.distance)).padStart(6, "0"), W * 0.06, H * 0.075);
    ctx.textAlign = "right";
    ctx.fillStyle = "#ffd34d";
    ctx.fillText("● " + this.coins, W * 0.94, H * 0.075);
    if (this.done) {
      ctx.fillStyle = "rgba(10,12,18,.62)"; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#fff"; ctx.textAlign = "center";
      ctx.font = "700 " + Math.round(H * 0.052) + "px ui-monospace,monospace";
      ctx.fillText("CRASHED", W / 2, H * 0.47);
      ctx.font = "500 " + Math.round(H * 0.030) + "px ui-monospace,monospace";
      ctx.fillText(Math.floor(this.distance) + " m", W / 2, H * 0.55);
    }
  };

  Runner.ACTIONS = ["none", "left", "right", "jump", "roll"];
  Runner.RETINA_W = RETINA_W;
  Runner.RETINA_H = RETINA_H;
  global.Runner = Runner;
})(window);
