/* Live spiking simulation of a Drosophila visual system, in the browser.
 *
 * This is not a replay. Every frame the retina is rendered from the game state,
 * injected into the photoreceptors, and the network is stepped; the action that
 * comes out is whatever the descending readout says this time. Run it twice and
 * it plays differently.
 *
 * Connectivity is source-major CSR: rows[i]..rows[i+1] are neuron i's outgoing
 * edges, with int32 targets in one block and int8 weights in another. Each step
 * touches only the rows of neurons that actually fired, so the cost tracks the
 * firing rate rather than the 3.7M synapses.
 */
(function (global) {
  "use strict";

  function FlyBrain(spec, targets, weights, rows) {
    this.spec = spec;
    this.n = spec.n;
    this.targets = targets;          // Int32Array, nnz
    this.weights = weights;          // Int8Array, nnz
    this.rows = rows;                // Int32Array, n+1
    this.wscale = spec.scale / 127.0;

    this.v = new Float32Array(this.n);
    this.trace = new Float32Array(this.n);
    this.spikes = new Uint8Array(this.n);
    this.refrac = new Int16Array(this.n);
    this.acc = new Float32Array(this.n);

    this.vth = Float32Array.from(spec.v_th);
    this.bias = Float32Array.from(spec.bias);
    this.inGain = Float32Array.from(spec.in_gain);
    this.photo = Int32Array.from(spec.photo);
    this.photoPixel = Int32Array.from(spec.photo_pixel);
    this.ext = new Float32Array(this.n);
    this.prevPx = null;

    this.decay = spec.decay;
    this.gain = spec.gain;
    this.refractory = spec.refractory | 0;
    this.traceDecay = spec.trace_decay;
    this.stepsPerFrame = spec.steps_per_frame;

    // Readout, if a trained policy shipped with the network.
    var p = spec.policy || {};
    this.hasPolicy = !!(p.read && p.read.length);
    if (this.hasPolicy) {
      this.read = Int32Array.from(p.read);
      this.nAct = p.n_actions;
      this.pW = Float32Array.from(p.W);     // (read+1) x nAct, row-major
      this.pMu = Float32Array.from(p.mu);
      this.pSd = Float32Array.from(p.sd);
    }
    this.lastSpikeCount = 0;
  }

  FlyBrain.prototype.reset = function () {
    this.v.fill(0); this.trace.fill(0);
    this.spikes.fill(0); this.refrac.fill(0);
    this.prevPx = null;
  };

  /* One timestep. Hot loop: keep it monomorphic and branch-light. */
  FlyBrain.prototype.step = function () {
    var n = this.n, acc = this.acc, rows = this.rows,
        tg = this.targets, wt = this.weights, sp = this.spikes;
    acc.fill(0);

    // Scatter the outgoing weights of everything that fired last step.
    for (var i = 0; i < n; i++) {
      if (sp[i] === 0) continue;
      var a = rows[i], b = rows[i + 1];
      for (var k = a; k < b; k++) acc[tg[k]] += wt[k];
    }

    var v = this.v, vth = this.vth, bias = this.bias, ig = this.inGain,
        ext = this.ext, refr = this.refrac, tr = this.trace,
        g = this.gain * this.wscale, d = this.decay, td = this.traceDecay,
        rf = this.refractory, fired = 0;

    for (var j = 0; j < n; j++) {
      var I = acc[j] * g * ig[j] + bias[j] + ext[j];
      var vj = v[j] * d + I;
      var free = refr[j] <= 0;
      if (!free) { vj = 0; refr[j]--; }
      var s = (free && vj >= vth[j]) ? 1 : 0;
      if (s) { vj = 0; refr[j] = rf; fired++; }
      v[j] = vj;
      sp[j] = s;
      tr[j] = tr[j] * td + s;
    }
    this.lastSpikeCount = fired;
    return fired;
  };

  /* Drive the photoreceptors from a retina frame and advance one game frame.
   * Photoreceptors are histaminergic and light-inverted, and respond to
   * contrast and change rather than absolute level -- hence the sign and the
   * frame difference. */
  FlyBrain.prototype.perceive = function (retina) {
    var ph = this.photo, pp = this.photoPixel, ext = this.ext;
    var inp = this.spec.input;
    var m = ph.length, px = new Float32Array(m), mean = 0;
    for (var i = 0; i < m; i++) { var val = retina[pp[i]]; px[i] = val; mean += val; }
    mean /= m;
    var prev = this.prevPx;
    ext.fill(0);
    for (var k = 0; k < m; k++) {
      var motion = prev ? (prev[k] - px[k]) : 0;
      ext[ph[k]] = inp.gain * (inp.sustained * (mean - px[k]) + inp.motion * motion);
    }
    this.prevPx = px;
    for (var s = 0; s < this.stepsPerFrame; s++) this.step();
  };

  /* Action from the trained readout over the populations that carry the signal. */
  FlyBrain.prototype.decide = function () {
    if (!this.hasPolicy) return (Math.random() * 5) | 0;
    var r = this.read, tr = this.trace, mu = this.pMu, sd = this.pSd,
        W = this.pW, A = this.nAct, m = r.length;
    var best = 0, bestv = -Infinity;
    for (var a = 0; a < A; a++) {
      var acc = W[m * A + a];                 // bias row
      for (var i = 0; i < m; i++) {
        acc += ((tr[r[i]] - mu[i]) / sd[i]) * W[i * A + a];
      }
      if (acc > bestv) { bestv = acc; best = a; }
    }
    return best;
  };

  FlyBrain.prototype.activity = function (idx) {
    var tr = this.trace, c = 0;
    for (var i = 0; i < idx.length; i++) if (tr[idx[i]] > 0) c++;
    return c / idx.length;
  };

  /* ------------------------------------------------------------------ load */
  FlyBrain.load = function (base, onProgress) {
    base = base || "";
    return fetch(base + "net.json").then(function (r) {
      if (!r.ok) throw new Error("net.json HTTP " + r.status);
      return r.json();
    }).then(function (spec) {
      var jobs = [fetch(base + "rows.bin").then(function (r) { return r.arrayBuffer(); })];
      for (var f = 0; f < spec.files; f++) {
        jobs.push(fetch(base + "net-" + f + ".bin").then(function (r) {
          if (!r.ok) throw new Error("weights HTTP " + r.status);
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
          blob.set(new Uint8Array(bufs[i]), off);
          off += bufs[i].byteLength;
        }
        var nnz = spec.nnz;
        // Targets occupy the first 4*nnz bytes, weights the remaining nnz.
        var targets = new Int32Array(blob.buffer.slice(0, nnz * 4));
        var weights = new Int8Array(blob.buffer.slice(nnz * 4, nnz * 5));
        return new FlyBrain(spec, targets, weights, rows);
      });
    });
  };

  global.FlyBrain = FlyBrain;
})(window);
