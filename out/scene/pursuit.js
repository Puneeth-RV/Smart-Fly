/* Two flies, two brains, one arena.
 *
 * Each fly runs its own copy of the connectome's visual system and sees the
 * other fly with its own eyes. Nothing is scripted: fly A turns because its
 * right eye is more excited than its left, and fly B flees because the thing on
 * its retina is growing.
 *
 * Both share a single copy of the weights -- the connectome is the species, not
 * the individual -- and keep their own membrane voltages, spike traces and
 * refractory counters. A second fly costs state, not another 25 MB.
 *
 * Steering is a bilateral comparison across the visual projection neurons (the
 * LC cells, the optic lobe's object-detection output): more right than left,
 * turn right. That is how the optomotor response and visual pursuit work in the
 * animal, and it needs no trained decoder -- which matters here, because the
 * thing this network was measured to be bad at is deep fine-grained decoding,
 * and this is neither deep nor fine.
 */
(function (global) {
  "use strict";

  var RW = 15, RH = 24;             // retina the connectome was wired against
  var FOV = 2.44;                   // ~140 deg per eye, overlapping at the front

  /* ---------------------------------------------------------------- brain */
  function Brain(shared) {
    var n = shared.n;
    this.s = shared;
    this.n = n;
    this.v = new Float32Array(n);
    this.trace = new Float32Array(n);
    this.spikes = new Uint8Array(n);
    this.refrac = new Int16Array(n);
    this.acc = new Float32Array(n);
    this.ext = new Float32Array(n);
    this.prevPx = null;
    this.spikeCount = 0;
  }

  Brain.prototype.step = function () {
    var s = this.s, n = this.n, acc = this.acc, rows = s.rows,
        tg = s.targets, wt = s.weights, sp = this.spikes;
    acc.fill(0);
    for (var i = 0; i < n; i++) {
      if (sp[i] === 0) continue;
      var a = rows[i], b = rows[i + 1];
      for (var k = a; k < b; k++) acc[tg[k]] += wt[k];
    }
    var v = this.v, tr = this.trace, refr = this.refrac, ext = this.ext,
        vth = s.vth, bias = s.bias, ig = s.inGain,
        g = s.gain * s.wscale, d = s.decay, td = s.traceDecay, rf = s.refractory,
        fired = 0;
    for (var j = 0; j < n; j++) {
      var I = acc[j] * g * ig[j] + bias[j] + ext[j];
      var vj = v[j] * d + I;
      var free = refr[j] <= 0;
      if (!free) { vj = 0; refr[j]--; }
      var fire = (free && vj >= vth[j]) ? 1 : 0;
      if (fire) { vj = 0; refr[j] = rf; fired++; }
      v[j] = vj; sp[j] = fire; tr[j] = tr[j] * td + fire;
    }
    this.spikeCount = fired;
  };

  /* Paint what this fly can see of the other one, per eye, and inject it.
   * The retina is the same 15x24 field the network was built around; the left
   * half belongs to the left eye, the right half to the right. */
  Brain.prototype.see = function (bearing, angSize, closing) {
    var s = this.s;
    var img = s.scratch; img.fill(0);
    // bearing: radians, 0 straight ahead, positive to the fly's right.
    var halfW = RW / 2;
    var u = halfW + (bearing / FOV) * RW;      // horizontal position on retina
    var r = Math.max(0.9, angSize / FOV * RW * 0.5);
    if (u > -r && u < RW + r) {
      var cy = RH * 0.52;
      var ry = Math.max(1.1, r * 1.35);
      for (var y = 0; y < RH; y++) {
        var dy = (y - cy) / ry;
        if (dy * dy > 1) continue;
        var halfSpan = r * Math.sqrt(1 - dy * dy);
        var x0 = Math.max(0, Math.floor(u - halfSpan));
        var x1 = Math.min(RW - 1, Math.ceil(u + halfSpan));
        for (var x = x0; x <= x1; x++) img[y * RW + x] = 1.0;
      }
    }
    // Drive photoreceptors: inverted (they are inhibitory and light-inverted),
    // contrast plus a transient term, exactly as offline.
    var ph = s.photo, pp = s.photoPixel, eye = s.photoEye, inp = s.input;
    var m = ph.length, px = s.pxBuf, mean = 0;
    for (var i = 0; i < m; i++) {
      var val = img[pp[i]];
      // An eye only sees its own side of the field.
      if (eye[i] < 0 && pp[i] % RW >= halfW + 2) val = 0;
      if (eye[i] > 0 && pp[i] % RW < halfW - 2) val = 0;
      px[i] = val; mean += val;
    }
    mean /= m;
    var prev = this.prevPx, ext = this.ext;
    ext.fill(0);
    for (var k = 0; k < m; k++) {
      var motion = prev ? (prev[k] - px[k]) : 0;
      ext[ph[k]] = inp.gain * (inp.sustained * (mean - px[k]) + inp.motion * motion);
    }
    if (!this.prevPx) this.prevPx = new Float32Array(m);
    this.prevPx.set(px);
    for (var t = 0; t < s.stepsPerFrame; t++) this.step();
    this.retina = img;
  };

  /* Bilateral readout: the only thing the body is told. */
  Brain.prototype.command = function () {
    var s = this.s, tr = this.trace;
    var L = 0, R = 0, i;
    for (i = 0; i < s.vpL.length; i++) L += tr[s.vpL[i]];
    for (i = 0; i < s.vpR.length; i++) R += tr[s.vpR[i]];
    L /= s.vpL.length; R /= s.vpR.length;
    var total = L + R;
    var turn = total > 1e-6 ? (R - L) / total : 0;   // -1 left .. +1 right
    return { turn: turn, drive: total, L: L, R: R };
  };

  /* ----------------------------------------------------------------- body */
  function Body(x, z, heading) {
    this.x = x; this.z = z; this.h = heading;
    this.speed = 0; this.turnRate = 0;
  }
  Body.prototype.advance = function (turn, forward, dt, maxTurn, maxSpeed) {
    this.turnRate += (turn * maxTurn - this.turnRate) * 0.35;
    this.h += this.turnRate * dt;
    this.speed += (forward * maxSpeed - this.speed) * 0.25;
    this.x += Math.sin(this.h) * this.speed * dt;
    this.z += Math.cos(this.h) * this.speed * dt;
    var R = 5.2;                                  // arena radius, soft wall
    var d = Math.hypot(this.x, this.z);
    if (d > R) {
      var k = R / d; this.x *= k; this.z *= k;
      this.h += 0.5;                              // turn away from the wall
    }
  };

  /* Where is `other` in my visual field, and how big? */
  function relativeView(me, other) {
    var dx = other.x - me.x, dz = other.z - me.z;
    var dist = Math.hypot(dx, dz) || 1e-5;
    var world = Math.atan2(dx, dz);
    var bearing = world - me.h;
    while (bearing > Math.PI) bearing -= 2 * Math.PI;
    while (bearing < -Math.PI) bearing += 2 * Math.PI;
    return { bearing: bearing, dist: dist, angSize: 2 * Math.atan(0.22 / dist) };
  }

  /* --------------------------------------------------------------- loader */
  function loadShared(base, onProgress) {
    base = base || "";
    return fetch(base + "net.json").then(function (r) {
      if (!r.ok) throw new Error("net.json HTTP " + r.status);
      return r.json();
    }).then(function (spec) {
      var jobs = [fetch(base + "rows.bin").then(function (r) { return r.arrayBuffer(); })];
      for (var f = 0; f < spec.files; f++) {
        jobs.push(fetch(base + "net-" + f + ".bin").then(function (r) {
          if (onProgress) onProgress();
          return r.arrayBuffer();
        }));
      }
      return Promise.all(jobs).then(function (bufs) {
        var rows = new Int32Array(bufs[0]);
        var total = 0, i;
        for (i = 1; i < bufs.length; i++) total += bufs[i].byteLength;
        var blob = new Uint8Array(total), off = 0;
        for (i = 1; i < bufs.length; i++) {
          blob.set(new Uint8Array(bufs[i]), off); off += bufs[i].byteLength;
        }
        var nnz = spec.nnz;
        var shared = {
          spec: spec, n: spec.n, rows: rows,
          targets: new Int32Array(blob.buffer.slice(0, nnz * 4)),
          weights: new Int8Array(blob.buffer.slice(nnz * 4, nnz * 5)),
          wscale: spec.scale / 127,
          vth: Float32Array.from(spec.v_th),
          bias: Float32Array.from(spec.bias),
          inGain: Float32Array.from(spec.in_gain),
          photo: Int32Array.from(spec.photo),
          photoPixel: Int32Array.from(spec.photo_pixel),
          photoEye: Int8Array.from(spec.photo_eye),
          vpL: Int32Array.from(spec.vp_L),
          vpR: Int32Array.from(spec.vp_R),
          decay: spec.decay, gain: spec.gain, refractory: spec.refractory | 0,
          traceDecay: spec.trace_decay, stepsPerFrame: spec.steps_per_frame,
          input: spec.input
        };
        shared.scratch = new Float32Array(RW * RH);
        shared.pxBuf = new Float32Array(shared.photo.length);
        return shared;
      });
    });
  }

  global.Pursuit = {
    loadShared: loadShared, Brain: Brain, Body: Body,
    relativeView: relativeView, RW: RW, RH: RH
  };
})(window);
