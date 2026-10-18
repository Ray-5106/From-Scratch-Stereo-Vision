// ---------------------------------------------------------------------
//  stereonet_infer.js
//
//  Inference-only port of stereonet.py's forward pass, for running the
//  trained network in a browser. Training, backpropagation and every
//  weight value come from the pure-Python model; this file only
//  evaluates the same arithmetic on the same weights. It is checked
//  against the Python reference by verify.js, and must agree to
//  floating-point noise or it does not ship.
//
//  Layout conventions match the Python exactly: flat arrays, channel
//  planar [c][y][x], weights [oc][ic][ky][kx] (and [oc][ic][kd][ky][kx]
//  for 3D), half-pixel bilinear sampling, Python's int() truncation.
// ---------------------------------------------------------------------
(function (root) {
  'use strict';

  function lrelu(x, s) {
    const y = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) { const v = x[i]; y[i] = v > 0 ? v : v * s; }
    return y;
  }

  function conv2d(x, ic, H, W, w, b, oc, k, s, p, d) {
    const eff = d * (k - 1) + 1;
    const OH = Math.floor((H + 2 * p - eff) / s) + 1;
    const OW = Math.floor((W + 2 * p - eff) / s) + 1;
    const y = new Float64Array(oc * OH * OW);
    const HW = H * W, kk = k * k, ickk = ic * kk;
    for (let o = 0; o < oc; o++) {
      const wo = o * ickk, ob = o * OH * OW, bias = b[o];
      for (let oy = 0; oy < OH; oy++) {
        const iy0 = oy * s - p, orow = ob + oy * OW;
        for (let ox = 0; ox < OW; ox++) {
          const ix0 = ox * s - p;
          let acc = bias;
          for (let c = 0; c < ic; c++) {
            const xb = c * HW, wc = wo + c * kk;
            for (let ky = 0; ky < k; ky++) {
              const iy = iy0 + ky * d;
              if (iy < 0 || iy >= H) continue;
              const xr = xb + iy * W, wr = wc + ky * k;
              for (let kx = 0; kx < k; kx++) {
                const ix = ix0 + kx * d;
                if (ix < 0 || ix >= W) continue;
                acc += x[xr + ix] * w[wr + kx];
              }
            }
          }
          y[orow + ox] = acc;
        }
      }
    }
    return { y, OH, OW };
  }

  function conv3d(x, ic, D, H, W, w, b, oc) {
    const k = 3, p = 1, HW = H * W, DHW = D * HW, kkk = 27, kk = 9, ickkk = ic * kkk;
    const y = new Float64Array(oc * DHW);
    for (let o = 0; o < oc; o++) {
      const wo = o * ickkk, ob = o * DHW, bias = b[o];
      for (let od = 0; od < D; od++) {
        const d0 = od - p;
        for (let oy = 0; oy < H; oy++) {
          const y0 = oy - p, orow = ob + od * HW + oy * W;
          for (let ox = 0; ox < W; ox++) {
            const x0 = ox - p;
            let acc = bias;
            for (let c = 0; c < ic; c++) {
              const xb = c * DHW, wc = wo + c * kkk;
              for (let kd = 0; kd < k; kd++) {
                const dd = d0 + kd;
                if (dd < 0 || dd >= D) continue;
                const xd = xb + dd * HW, wd = wc + kd * kk;
                for (let ky = 0; ky < k; ky++) {
                  const yy = y0 + ky;
                  if (yy < 0 || yy >= H) continue;
                  const xr = xd + yy * W, wr = wd + ky * k;
                  for (let kx = 0; kx < k; kx++) {
                    const xx = x0 + kx;
                    if (xx < 0 || xx >= W) continue;
                    acc += x[xr + xx] * w[wr + kx];
                  }
                }
              }
            }
            y[orow + ox] = acc;
          }
        }
      }
    }
    return y;
  }

  const INVALID_COST = 2.0, PENALTY = 8.0;

  function costVolume(fl, fr, C, H, W, D) {
    const HW = H * W, DHW = D * HW, vol = new Float64Array(C * DHW);
    for (let c = 0; c < C; c++) {
      const cb = c * HW, vb = c * DHW;
      for (let d = 0; d < D; d++) {
        const db = vb + d * HW;
        for (let y = 0; y < H; y++) {
          const row = y * W;
          for (let x = 0; x < W; x++) {
            const idx = db + row + x, xr = x - d;
            if (xr < 0) { vol[idx] = INVALID_COST; continue; }
            const v = fl[cb + row + x] - fr[cb + row + xr];
            vol[idx] = v < 0 ? -v : v;
          }
        }
      }
    }
    return vol;
  }

  function softArgmin(cost, H, W, D) {
    const HW = H * W, c = Float64Array.from(cost);
    for (let d = 1; d < D; d++)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < d && x < W; x++) c[d * HW + y * W + x] += PENALTY;
    const disp = new Float64Array(HW), e = new Float64Array(D);
    for (let i = 0; i < HW; i++) {
      let best = -c[i];
      for (let d = 1; d < D; d++) { const v = -c[d * HW + i]; if (v > best) best = v; }
      let tot = 0;
      for (let d = 0; d < D; d++) { e[d] = Math.exp(-c[d * HW + i] - best); tot += e[d]; }
      let acc = 0;
      for (let d = 0; d < D; d++) acc += d * (e[d] / tot);
      disp[i] = acc;
    }
    return disp;
  }

  function bilinear(x, C, H, W, H2, W2) {
    const sy = H / H2, sx = W / W2, out = new Float64Array(C * H2 * W2), HW = H * W;
    const taps = [];
    for (let oy = 0; oy < H2; oy++) {
      let fy = (oy + 0.5) * sy - 0.5; if (fy < 0) fy = 0;
      let y0 = Math.trunc(fy); if (y0 > H - 1) y0 = H - 1;
      const y1 = y0 + 1 < H ? y0 + 1 : y0, wy = fy - y0;
      for (let ox = 0; ox < W2; ox++) {
        let fx = (ox + 0.5) * sx - 0.5; if (fx < 0) fx = 0;
        let x0 = Math.trunc(fx); if (x0 > W - 1) x0 = W - 1;
        const x1 = x0 + 1 < W ? x0 + 1 : x0, wx = fx - x0;
        taps.push(oy * W2 + ox, y0 * W + x0, (1 - wy) * (1 - wx), y0 * W + x1,
                  (1 - wy) * wx, y1 * W + x0, wy * (1 - wx), y1 * W + x1, wy * wx);
      }
    }
    for (let c = 0; c < C; c++) {
      const cb = c * HW, ob = c * H2 * W2;
      for (let t = 0; t < taps.length; t += 9)
        out[ob + taps[t]] = x[cb + taps[t + 1]] * taps[t + 2] + x[cb + taps[t + 3]] * taps[t + 4] +
                            x[cb + taps[t + 5]] * taps[t + 6] + x[cb + taps[t + 7]] * taps[t + 8];
    }
    return out;
  }

  function avgpool2(img, H, W) {
    const H2 = Math.floor(H / 2), W2 = Math.floor(W / 2), out = new Float64Array(H2 * W2);
    for (let y = 0; y < H2; y++) {
      const r0 = 2 * y * W, r1 = r0 + W;
      for (let x = 0; x < W2; x++) {
        const x0 = 2 * x;
        out[y * W2 + x] = 0.25 * (img[r0 + x0] + img[r0 + x0 + 1] + img[r1 + x0] + img[r1 + x0 + 1]);
      }
    }
    return { out, H2, W2 };
  }

  // --- weights: the checkpoint's parameter blocks, in all_params order ---
  function buildNet(ckpt) {
    const cfg = ckpt.config, B = ckpt.blocks;
    let i = 0;
    const take = () => Float64Array.from(B[i++]);
    const net = { C: cfg.feat_C, Cc: cfg.cost_C, D: cfg.D, R: cfg.refine_C, dil: cfg.refine_dil };
    net.feat = [];
    for (let k = 0; k < 4; k++) net.feat.push({ w: take(), b: take() });
    net.filt = [];
    for (let k = 0; k < 3; k++) net.filt.push({ w: take(), b: take() });
    const ref = () => {
      const r = { convs: [] };
      for (let k = 0; k < net.dil.length; k++) r.convs.push({ w: take(), b: take() });
      r.last = { w: take(), b: take() };
      return r;
    };
    net.ref1 = ref(); net.ref2 = ref();
    if (i !== B.length) throw new Error('checkpoint has ' + B.length + ' blocks, net used ' + i);
    return net;
  }

  function features(net, img, H, W) {
    const C = net.C, f = net.feat;
    let r = conv2d(img, 1, H, W, f[0].w, f[0].b, C, 3, 2, 1, 1);
    let x = lrelu(r.y, 0.2), h = r.OH, w = r.OW;
    r = conv2d(x, C, h, w, f[1].w, f[1].b, C, 3, 2, 1, 1); x = lrelu(r.y, 0.2); h = r.OH; w = r.OW;
    r = conv2d(x, C, h, w, f[2].w, f[2].b, C, 3, 1, 1, 1); x = lrelu(r.y, 0.2);
    r = conv2d(x, C, h, w, f[3].w, f[3].b, C, 3, 1, 1, 1);
    return { f: r.y, h, w };
  }

  function refine(ref, R, dil, disp, guide, H, W) {
    const HW = H * W, x = new Float64Array(2 * HW);
    x.set(disp, 0); x.set(guide, HW);
    let y = x, ic = 2;
    for (let k = 0; k < ref.convs.length; k++) {
      const d = dil[k];
      y = lrelu(conv2d(y, ic, H, W, ref.convs[k].w, ref.convs[k].b, R, 3, 1, d, d).y, 0.2);
      ic = R;
    }
    const res = conv2d(y, ic, H, W, ref.last.w, ref.last.b, 1, 3, 1, 1, 1).y;
    const out = new Float64Array(HW);
    for (let i = 0; i < HW; i++) { const v = disp[i] + res[i]; out[i] = v > 0 ? v : 0; }
    return out;
  }

  function predict(net, left, right, W, H) {
    const gh = avgpool2(left, H, W);
    const L = features(net, left, H, W), Rt = features(net, right, H, W);
    const h = L.h, w = L.w;
    const vol = costVolume(L.f, Rt.f, net.C, h, w, net.D);
    let c = lrelu(conv3d(vol, net.C, net.D, h, w, net.filt[0].w, net.filt[0].b, net.Cc), 0.2);
    c = lrelu(conv3d(c, net.Cc, net.D, h, w, net.filt[1].w, net.filt[1].b, net.Cc), 0.2);
    c = conv3d(c, net.Cc, net.D, h, w, net.filt[2].w, net.filt[2].b, 1);
    const d0 = softArgmin(c, h, w, net.D);
    const H2 = Math.floor(H / 2), W2 = Math.floor(W / 2);
    const u1 = bilinear(d0, 1, h, w, H2, W2); for (let i = 0; i < u1.length; i++) u1[i] *= 2;
    const d1 = refine(net.ref1, net.R, net.dil, u1, gh.out, H2, W2);
    const u2 = bilinear(d1, 1, H2, W2, H, W); for (let i = 0; i < u2.length; i++) u2[i] *= 2;
    return refine(net.ref2, net.R, net.dil, u2, left, H, W);
  }

  function predictAtScale(net, left, right, W, H, ww) {
    if (ww >= W) { const d = predict(net, left, right, W, H); return { disp: d, Ww: W, Hw: H }; }
    const scale = ww / W, Ww = ww;
    let Hw = Math.trunc(H * scale + 0.5); if (Hw < 8) Hw = 8;
    const lw = bilinear(left, 1, H, W, Hw, Ww), rw = bilinear(right, 1, H, W, Hw, Ww);
    const dw = predict(net, lw, rw, Ww, Hw);
    const df = bilinear(dw, 1, Hw, Ww, H, W), inv = 1 / scale;
    for (let i = 0; i < df.length; i++) df[i] *= inv;
    return { disp: df, Ww, Hw };
  }

  function peak(disp) {
    const s = Float64Array.from(disp).sort();
    const v = s[Math.trunc(s.length * 0.95)];
    return v > 0.5 ? v : 0.5;
  }

  // Coarse-to-fine scale selection with self-correction; see predict_auto.
  function predictAuto(net, left, right, W, H, opts) {
    const T = (opts && opts.targetPeak) || 20, maxW = (opts && opts.maxWidth) || 320;
    const passes = (opts && opts.passes) || 3, log = (opts && opts.log) || (() => {});
    const pw = 64 < W ? 64 : W;
    let ph = Math.trunc(H * pw / W + 0.5); if (ph < 8) ph = 8;
    const lp = bilinear(left, 1, H, W, ph, pw), rp = bilinear(right, 1, H, W, ph, pw);
    let pk = peak(predict(net, lp, rp, pw, ph)) * (W / pw);
    const widthFor = (p) => {
      let w = Math.trunc(W * T / Math.max(p, 1e-6) + 0.5);
      if (w > W) w = W; if (w > maxW) w = maxW; if (w < 48) w = 48; return w;
    };
    let work = widthFor(pk);
    log('probe estimates peak disparity ' + pk.toFixed(1) + ' px, running at ' + work + ' px wide');
    let r = predictAtScale(net, left, right, W, H, work);
    for (let k = 0; k < passes - 1; k++) {
      const np = peak(r.disp), nw = widthFor(np);
      if (Math.abs(nw - work) <= 0.08 * work) break;
      log('re-measured peak ' + np.toFixed(1) + ' px, rescaling ' + work + ' -> ' + nw);
      work = nw; r = predictAtScale(net, left, right, W, H, work);
    }
    return r;
  }

  const api = { buildNet, predict, predictAtScale, predictAuto, bilinear, avgpool2 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.StereoNetInfer = api;
})(this);
