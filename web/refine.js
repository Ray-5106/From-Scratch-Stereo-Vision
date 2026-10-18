// Full-resolution band refinement, planes and background layer.
// A line-for-line port of SECTION 25 of stereonet.py; verify_refine.js
// checks it against the Python output.
(function (root) {
  'use strict';
  const trunc = Math.trunc;

  // census signature split over two 32-bit words (up to 62 bits)
  function census(img, W, H, rx, ry) {
    const lo = new Uint32Array(W * H), hi = new Uint32Array(W * H);
    const offs = [];
    for (let dy = -ry; dy <= ry; dy++) for (let dx = -rx; dx <= rx; dx++) if (dx || dy) offs.push([dx, dy]);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const c = img[y * W + x]; let a = 0, b = 0;
      for (let k = 0; k < offs.length; k++) {
        let xx = x + offs[k][0], yy = y + offs[k][1];
        if (xx < 0) xx = 0; else if (xx >= W) xx = W - 1;
        if (yy < 0) yy = 0; else if (yy >= H) yy = H - 1;
        if (img[yy * W + xx] < c) { if (k < 31) a |= (1 << k); else b |= (1 << (k - 31)); }
      }
      lo[y * W + x] = a >>> 0; hi[y * W + x] = b >>> 0;
    }
    return { lo, hi, nbits: offs.length };
  }
  function pop32(v) { v = v - ((v >>> 1) & 0x55555555); v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24; }

  function bandCosts(cl, cr, base, W, H, R, invalid) {
    const K = 2 * R + 1, C = new Float64Array(W * H * K).fill(invalid);
    for (let y = 0; y < H; y++) {
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const p = row + x, b = base[p] - R, al = cl.lo[p], ah = cl.hi[p], o = p * K;
        for (let k = 0; k < K; k++) {
          const xr = x - (b + k);
          if (b + k < 0 || xr < 0 || xr >= W) continue;
          const q = row + xr;
          C[o + k] = pop32((al ^ cr.lo[q]) >>> 0) + pop32((ah ^ cr.hi[q]) >>> 0);
        }
      }
    }
    return C;
  }

  function aggregatePath(C, base, W, H, K, P1, P2, dx, dy, S) {
    const N = W * H, L = new Float64Array(N * K), mins = new Float64Array(N);
    const y0 = dy >= 0 ? 0 : H - 1, y1 = dy >= 0 ? H : -1, ys = dy >= 0 ? 1 : -1;
    const x0 = dx >= 0 ? 0 : W - 1, x1 = dx >= 0 ? W : -1, xs = dx >= 0 ? 1 : -1;
    for (let y = y0; y !== y1; y += ys) for (let x = x0; x !== x1; x += xs) {
      const p = y * W + x, o = p * K, qx = x - dx, qy = y - dy;
      if (qx < 0 || qx >= W || qy < 0 || qy >= H) {
        let m = 1.0e30;
        for (let k = 0; k < K; k++) { const v = C[o + k]; L[o + k] = v; if (v < m) m = v; }
        mins[p] = m; continue;
      }
      const q = qy * W + qx, oq = q * K, mq = mins[q], shift = base[p] - base[q];
      let m = 1.0e30;
      for (let k = 0; k < K; k++) {
        const kq = k + shift; let best = mq + P2, v;
        if (0 <= kq && kq < K) { v = L[oq + kq]; if (v < best) best = v; }
        if (0 <= kq - 1 && kq - 1 < K) { v = L[oq + kq - 1] + P1; if (v < best) best = v; }
        if (0 <= kq + 1 && kq + 1 < K) { v = L[oq + kq + 1] + P1; if (v < best) best = v; }
        v = C[o + k] + best - mq; L[o + k] = v; if (v < m) m = v;
      }
      mins[p] = m;
    }
    for (let i = 0; i < N * K; i++) S[i] += L[i];
  }

  function refineFullres(left, right, disp0, W, H, o) {
    o = Object.assign({ R: 10, P1: 12, P2: 150, rx: 4, ry: 3 }, o || {});
    const R = o.R, K = 2 * R + 1, N = W * H, base = new Int32Array(N);
    for (let i = 0; i < N; i++) { const v = trunc(disp0[i] + 0.5); base[i] = v > 0 ? v : 0; }
    const cl = census(left, W, H, o.rx, o.ry), cr = census(right, W, H, o.rx, o.ry);
    const invalid = cl.nbits + 0 + 0 + 4.0;
    const C = bandCosts(cl, cr, base, W, H, R, invalid), S = new Float64Array(N * K);
    const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]];
    for (const [dx, dy] of dirs) aggregatePath(C, base, W, H, K, o.P1, o.P2, dx, dy, S);
    const out = new Float64Array(N);
    for (let p = 0; p < N; p++) {
      const oo = p * K; let bk = 0, bv = S[oo];
      for (let k = 1; k < K; k++) if (S[oo + k] < bv) { bv = S[oo + k]; bk = k; }
      if (bk === 0 || bk === K - 1) { out[p] = disp0[p]; continue; }
      const a = S[oo + bk - 1], b = S[oo + bk + 1], den = a - 2.0 * bv + b;
      let off = 0.0; if (den > 1.0e-9) off = 0.5 * (a - b) / den;
      out[p] = base[p] - R + bk + off;
    }
    return out;
  }

  function mirror(img, W, H) {
    const out = new Float64Array(W * H);
    for (let y = 0; y < H; y++) { const r = y * W; for (let x = 0; x < W; x++) out[r + x] = img[r + W - 1 - x]; }
    return out;
  }

  function leftToRight(dl, W, H) {
    const dr = new Float64Array(W * H).fill(-1.0);
    for (let y = 0; y < H; y++) {
      const r = y * W;
      for (let x = 0; x < W; x++) { const d = dl[r + x], xr = trunc(x - d + 0.5); if (0 <= xr && xr < W && d > dr[r + xr]) dr[r + xr] = d; }
      for (let x = 0; x < W; x++) if (dr[r + x] < 0.0) {
        let a = x - 1; while (a >= 0 && dr[r + a] < 0.0) a--;
        let b = x + 1; while (b < W && dr[r + b] < 0.0) b++;
        const va = a >= 0 ? dr[r + a] : 1.0e9, vb = b < W ? dr[r + b] : 1.0e9, v = va < vb ? va : vb;
        dr[r + x] = v < 1.0e8 ? v : 0.0;
      }
    }
    return dr;
  }

  function lrRefine(left, right, disp0, W, H, tol, o) {
    const dl = refineFullres(left, right, disp0, W, H, o);
    const d0r = leftToRight(disp0, W, H);
    const dr = mirror(refineFullres(mirror(right, W, H), mirror(left, W, H), mirror(d0r, W, H), W, H, o), W, H);
    const status = new Uint8Array(W * H), out = Float64Array.from(dl);
    for (let y = 0; y < H; y++) { const r = y * W; for (let x = 0; x < W; x++) {
      const d = dl[r + x], xr = trunc(x - d + 0.5);
      if (xr < 0 || xr >= W) { status[r + x] = 1; continue; }
      const e = dr[r + xr]; if (Math.abs(d - e) <= tol) continue;
      status[r + x] = e > d ? 1 : 2;
    } }
    for (let i = 0; i < W * H; i++) if (status[i] === 2) out[i] = disp0[i];
    for (let y = 0; y < H; y++) { const r = y * W; for (let x = 0; x < W; x++) {
      if (status[r + x] !== 1) continue;
      let a = x - 1; while (a >= 0 && status[r + a] === 1) a--;
      let b = x + 1; while (b < W && status[r + b] === 1) b++;
      const va = a >= 0 ? out[r + a] : 1.0e9, vb = b < W ? out[r + b] : 1.0e9, v = va < vb ? va : vb;
      out[r + x] = v < 1.0e8 ? v : disp0[r + x];
    } }
    return { disp: out, status };
  }

  function slic(rgb, W, H, step, compactness, iters) {
    let cs = [];
    for (let y = trunc(step / 2); y < H; y += step) for (let x = trunc(step / 2); x < W; x += step) {
      const i = y * W + x; cs.push([rgb[3 * i] * 255.0, rgb[3 * i + 1] * 255.0, rgb[3 * i + 2] * 255.0, x, y]);
    }
    const N = W * H, lab = new Int32Array(N).fill(-1), m2 = (compactness / step) ** 2;
    for (let it = 0; it < iters; it++) {
      const dist = new Float64Array(N).fill(1.0e30);
      for (let ci = 0; ci < cs.length; ci++) {
        const [cr, cg, cb, cx, cy] = cs[ci];
        let x0 = trunc(cx) - step, x1 = trunc(cx) + step, y0 = trunc(cy) - step, y1 = trunc(cy) + step;
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 >= W) x1 = W - 1; if (y1 >= H) y1 = H - 1;
        for (let yy = y0; yy <= y1; yy++) { const r = yy * W, dy = yy - cy;
          for (let xx = x0; xx <= x1; xx++) { const i = r + xx;
            const a = rgb[3 * i] * 255.0 - cr, b = rgb[3 * i + 1] * 255.0 - cg, c = rgb[3 * i + 2] * 255.0 - cb, dx = xx - cx;
            const dd = a * a + b * b + c * c + m2 * (dx * dx + dy * dy);
            if (dd < dist[i]) { dist[i] = dd; lab[i] = ci; }
          } }
      }
      const acc = cs.map(() => [0.0, 0.0, 0.0, 0, 0, 0]);
      for (let i = 0; i < N; i++) { const l = lab[i]; if (l < 0) continue; const a = acc[l];
        a[0] += rgb[3 * i]; a[1] += rgb[3 * i + 1]; a[2] += rgb[3 * i + 2]; a[3] += i % W; a[4] += Math.floor(i / W); a[5] += 1; }
      for (let ci = 0; ci < cs.length; ci++) { const a = acc[ci];
        if (a[5]) { const n = a[5]; cs[ci] = [a[0] / n * 255.0, a[1] / n * 255.0, a[2] / n * 255.0, a[3] / n, a[4] / n]; } }
    }
    return { lab, n: cs.length };
  }

  function solve3(A, b) {
    const [[a, bb, c], [d, e, f], [g, h, k]] = A;
    const det = a * (e * k - f * h) - bb * (d * k - f * g) + c * (d * h - e * g);
    if (Math.abs(det) < 1.0e-9) return null;
    const x = (b[0] * (e * k - f * h) - bb * (b[1] * k - f * b[2]) + c * (b[1] * h - e * b[2])) / det;
    const y = (a * (b[1] * k - f * b[2]) - b[0] * (d * k - f * g) + c * (d * b[2] - b[1] * g)) / det;
    const z = (a * (e * b[2] - b[1] * h) - bb * (d * b[2] - b[1] * g) + b[0] * (d * h - e * g)) / det;
    return [x, y, z];
  }
  function lsqPlane(xs, ys, ds) {
    const n = xs.length; let sx = 0.0, sy = 0.0, sd = 0.0;
    for (let i = 0; i < n; i++) sx += xs[i]; for (let i = 0; i < n; i++) sy += ys[i]; for (let i = 0; i < n; i++) sd += ds[i];
    const mx = sx / n, my = sy / n, md = sd / n; let sxx = 0, sxy = 0, syy = 0, sxd = 0, syd = 0;
    for (let i = 0; i < n; i++) { const u = xs[i] - mx, v = ys[i] - my, w = ds[i] - md;
      sxx += u * u; sxy += u * v; syy += v * v; sxd += u * w; syd += v * w; }
    const det = sxx * syy - sxy * sxy;
    if (Math.abs(det) < 1.0e-9) return [0.0, 0.0, md];
    const a = (sxd * syy - syd * sxy) / det, b = (syd * sxx - sxd * sxy) / det;
    return [a, b, md - a * mx - b * my];
  }

  function planeSnap(disp, rgb, W, H, valid, o) {
    o = Object.assign({ step: 12, thr: 1.0, minInl: 0.4, trials: 40, seed: 12345 }, o || {});
    const { lab, n: nseg } = slic(rgb, W, H, o.step, 10.0, 5);
    const groups = Array.from({ length: nseg }, () => []);
    for (let i = 0; i < W * H; i++) if (lab[i] >= 0) groups[lab[i]].push(i);
    const out = Float64Array.from(disp); let s = o.seed >>> 0;
    const fx = i => i % W, fy = i => Math.floor(i / W);
    for (const g of groups) {
      const pts = valid ? g.filter(i => valid[i]) : g; const n = pts.length;
      if (n < 12) continue;
      let best = null, bestc = -1;
      for (let t = 0; t < o.trials; t++) {
        const tri = [];
        for (let u = 0; u < 3; u++) {
          s = (s ^ ((s << 13) >>> 0)) >>> 0; s = (s ^ (s >>> 17)) >>> 0; s = (s ^ ((s << 5) >>> 0)) >>> 0;
          tri.push(pts[s % n]);
        }
        const pl = solve3(tri.map(i => [fx(i), fy(i), 1.0]), tri.map(i => disp[i]));
        if (!pl) continue;
        const [a, b, c] = pl; let cnt = 0;
        for (const i of pts) if (Math.abs(a * fx(i) + b * fy(i) + c - disp[i]) <= o.thr) cnt++;
        if (cnt > bestc) { bestc = cnt; best = pl; }
      }
      if (!best || bestc < o.minInl * n) continue;
      let [a, b, c] = best;
      const inl = pts.filter(i => Math.abs(a * fx(i) + b * fy(i) + c - disp[i]) <= o.thr);
      [a, b, c] = lsqPlane(inl.map(fx), inl.map(fy), inl.map(i => disp[i]));
      for (const i of g) { const v = a * fx(i) + b * fy(i) + c;
        if (Math.abs(v - disp[i]) <= o.thr || (valid && !valid[i])) out[i] = v > 0.0 ? v : 0.0; }
    }
    return out;
  }

  // one 3x3 median pass, same selection rule as median_filter in Python
  function median3(d, W, H) {
    const out = new Float64Array(W * H), v = [];
    for (let y = 0; y < H; y++) { const y0 = y - 1 > 0 ? y - 1 : 0, y1 = y + 1 < H - 1 ? y + 1 : H - 1;
      for (let x = 0; x < W; x++) { const x0 = x - 1 > 0 ? x - 1 : 0, x1 = x + 1 < W - 1 ? x + 1 : W - 1;
        v.length = 0; for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) v.push(d[yy * W + xx]);
        v.sort((a, b) => a - b); out[y * W + x] = v[v.length >> 1]; } }
    return out;
  }

  function refineDisparity(left, right, rgb, disp0, W, H, log) {
    const r = lrRefine(left, right, disp0, W, H, 2.0, {});
    if (log) { let a = 0, b = 0; for (const s of r.status) { if (s === 1) a++; else if (s === 2) b++; }
      log('full-resolution refinement: ' + (100 * a / (W * H)).toFixed(1) + '% occluded, ' + (100 * b / (W * H)).toFixed(1) + '% mismatched'); }
    const valid = Array.from(r.status, s => s === 0);
    return { disp: median3(planeSnap(r.disp, rgb, W, H, valid, {}), W, H), status: r.status };
  }

  function minFilterArg(v, W, H, r, rv) {
    if (rv === undefined) rv = r;
    const N = W * H, rowv = new Float64Array(N), ri = new Int32Array(N), out = new Float64Array(N), oi = new Int32Array(N);
    for (let y = 0; y < H; y++) { const row = y * W;
      for (let x = 0; x < W; x++) { const a = x - r > 0 ? x - r : 0, b = x + r < W - 1 ? x + r : W - 1;
        let bi = row + a, bv = v[bi]; for (let xx = a + 1; xx <= b; xx++) if (v[row + xx] < bv) { bv = v[row + xx]; bi = row + xx; }
        rowv[row + x] = bv; ri[row + x] = bi; } }
    if (rv === 0) return { m: rowv, src: ri };
    for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) { const a = y - rv > 0 ? y - rv : 0, b = y + rv < H - 1 ? y + rv : H - 1;
      let bi = a * W + x, bv = rowv[bi]; for (let yy = a + 1; yy <= b; yy++) { const j = yy * W + x; if (rowv[j] < bv) { bv = rowv[j]; bi = j; } }
      out[y * W + x] = bv; oi[y * W + x] = ri[bi]; }
    return { m: out, src: oi };
  }

  function smoothBackground(m, disp, W, H, r) {
    const N = W * H, t = new Float64Array(N), out = new Float64Array(N);
    for (let y = 0; y < H; y++) { const row = y * W;
      for (let x = 0; x < W; x++) { const a = x - r > 0 ? x - r : 0, b = x + r < W - 1 ? x + r : W - 1;
        let s = 0.0; for (let xx = a; xx <= b; xx++) s += m[row + xx]; t[row + x] = s / (b - a + 1); } }
    for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) { const a = y - r > 0 ? y - r : 0, b = y + r < H - 1 ? y + r : H - 1;
      let s = 0.0; for (let yy = a; yy <= b; yy++) s += t[yy * W + x];
      const v = s / (b - a + 1), i = y * W + x; out[i] = v < disp[i] ? v : disp[i]; }
    return out;
  }

  // Region behind foreground edges: {mask, m (background disparity), src}
  function backgroundRegion(disp, W, H, radius, jumpRatio, minJump) {
    radius = radius || (W >= 64 ? trunc(W / 16) : 4); jumpRatio = jumpRatio || 0.15; minJump = minJump || 2.0;
    const mf = minFilterArg(disp, W, H, radius, 0), src = mf.src, N = W * H;
    const m = smoothBackground(mf.m, disp, W, H, radius >= 2 ? Math.floor(radius / 2) : 1);
    let mask = new Uint8Array(N);
    for (let i = 0; i < N; i++) { const d = disp[i]; let thr = jumpRatio * d; if (thr < minJump) thr = minJump; if (d - m[i] > thr) mask[i] = 1; }
    for (let it = 0; it < 2; it++) { const g = Uint8Array.from(mask);
      for (let y = 0; y < H; y++) { const row = y * W; for (let x = 0; x < W; x++) { if (mask[row + x]) continue;
        if ((x > 0 && mask[row + x - 1]) || (x < W - 1 && mask[row + x + 1]) || (y > 0 && mask[row - W + x]) || (y < H - 1 && mask[row + W + x])) g[row + x] = 1; } }
      mask = g; }
    return { mask, m, src };
  }

  const api = { smoothBackground, census, refineFullres, lrRefine, leftToRight, slic, planeSnap, refineDisparity, minFilterArg, backgroundRegion };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Refine = api;
})(this);
