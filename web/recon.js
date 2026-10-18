// ---------------------------------------------------------------------
//  recon.js -- disparity map to a 3D mesh, ported from stereonet.py
//  sections 11, 12, 22 and 24: left-margin repair, median and
//  image-guided bilateral filtering, depth clamp, unprojection,
//  adaptive vertex selection, Bowyer-Watson Delaunay, and the three
//  culls that keep occlusion boundaries from turning into spikes.
// ---------------------------------------------------------------------
(function (root) {
  'use strict';

  function fixLeftBorder(d, W, H, border) {
    border = Math.min(border, Math.floor(W / 2));
    const out = Float64Array.from(d);
    for (let y = 0; y < H; y++) {
      const a = d[y * W + border];
      for (let x = 0; x < border; x++) out[y * W + x] = a;
    }
    return out;
  }

  function median3(d, W, H, passes) {
    let cur = d;
    const v = new Float64Array(9);
    for (let p = 0; p < passes; p++) {
      const out = new Float64Array(W * H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let n = 0;
        for (let yy = Math.max(0, y - 1); yy <= Math.min(H - 1, y + 1); yy++)
          for (let xx = Math.max(0, x - 1); xx <= Math.min(W - 1, x + 1); xx++) v[n++] = cur[yy * W + xx];
        const s = Array.from(v.subarray(0, n)).sort((a, b) => a - b);
        out[y * W + x] = s[Math.floor(n / 2)];
      }
      cur = out;
    }
    return cur;
  }

  function bilateral(d, guide, W, H, r, ss, sc, passes) {
    const span = 2 * r + 1, sw = new Float64Array(span * span);
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++)
      sw[(dy + r) * span + dx + r] = Math.exp(-(dx * dx + dy * dy) / (2 * ss * ss));
    const T = 256, ct = new Float64Array(T);
    for (let i = 0; i < T; i++) { const t = i / (T - 1); ct[i] = Math.exp(-t * t / (2 * sc * sc)); }
    let cur = d;
    for (let p = 0; p < passes; p++) {
      const out = new Float64Array(W * H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const g = guide[y * W + x]; let acc = 0, ws = 0;
        for (let dy = Math.max(-r, -y); dy <= Math.min(r, H - 1 - y); dy++) {
          const row = (y + dy) * W, srow = (dy + r) * span + r;
          for (let dx = Math.max(-r, -x); dx <= Math.min(r, W - 1 - x); dx++) {
            const i = row + x + dx; let c = Math.abs(g - guide[i]);
            let ci = Math.trunc(c * (T - 1)); if (ci > T - 1) ci = T - 1;
            const w = sw[srow + dx] * ct[ci]; acc += w * cur[i]; ws += w;
          }
        }
        out[y * W + x] = ws > 0 ? acc / ws : cur[y * W + x];
      }
      cur = out;
    }
    return cur;
  }

  function circum(ax, ay, bx, by, cx, cy) {
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(d) < 1e-12) return null;
    const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    return [ux, uy, (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy)];
  }

  function delaunay(pts) {
    const n = pts.length; if (n < 3) return [];
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (const [x, y] of pts) { minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
    const span = Math.max(maxx - minx, maxy - miny) || 1, mx = (minx + maxx) / 2, my = (miny + maxy) / 2, m = span * 20;
    const V = pts.slice(); V.push([mx - m, my - span], [mx, my + m], [mx + m, my - span]);
    const mk = (i, j, k) => { const c = circum(V[i][0], V[i][1], V[j][0], V[j][1], V[k][0], V[k][1]); return c ? [i, j, k, c[0], c[1], c[2]] : null; };
    let tris = [mk(n, n + 1, n + 2)];
    for (let p = 0; p < n; p++) {
      const [px, py] = V[p], bad = [], keep = [];
      for (const t of tris) { const dx = px - t[3], dy = py - t[4]; (dx * dx + dy * dy < t[5] - 1e-12 ? bad : keep).push(t); }
      if (!bad.length) continue;
      const edges = new Map();
      for (const t of bad) for (const [a, b] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
        const key = a < b ? a * 1e6 + b : b * 1e6 + a; edges.set(key, (edges.get(key) || 0) + 1);
      }
      tris = keep;
      for (const [key, cnt] of edges) if (cnt === 1) {
        const a = Math.floor(key / 1e6), b = key % 1e6, t = mk(a, b, p); if (t) tris.push(t);
      }
    }
    return tris.filter(t => t[0] < n && t[1] < n && t[2] < n).map(t => [t[0], t[1], t[2]]);
  }

  function selectPoints(disp, W, H, target) {
    const grad = new Float64Array(W * H);
    for (let y = 0; y < H - 1; y++) for (let x = 0; x < W - 1; x++) {
      const i = y * W + x, gx = disp[i + 1] - disp[i], gy = disp[i + W] - disp[i];
      grad[i] = Math.sqrt(gx * gx + gy * gy);
    }
    const nEdge = Math.trunc(target * 0.45), nGrid = target - nEdge;
    const chosen = [], taken = new Uint8Array(W * H);
    const add = (x, y) => { const i = y * W + x; if (taken[i]) return false; taken[i] = 1; chosen.push([x, y]); return true; };
    for (const [x, y] of [[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]]) { taken[y * W + x] = 1; chosen.push([x, y]); }
    const step = Math.sqrt(W * H / Math.max(1, nGrid));
    for (let gy = 0; gy < H; gy += step) for (let gx = 0; gx < W; gx += step)
      add(Math.min(W - 1, Math.trunc(gx)), Math.min(H - 1, Math.trunc(gy)));
    const ranked = [];
    for (let i = 0; i < W * H; i++) if (grad[i] > 0 && !taken[i]) ranked.push(i);
    ranked.sort((a, b) => grad[b] - grad[a]);
    const placed = [];
    for (const i of ranked) {
      if (placed.length >= nEdge) break;
      const x = i % W, y = Math.floor(i / W); let ok = true;
      for (let k = Math.max(0, placed.length - 40); k < placed.length; k++)
        if (Math.abs(placed[k][0] - x) < 1.5 && Math.abs(placed[k][1] - y) < 1.5) { ok = false; break; }
      if (ok && add(x, y)) placed.push([x, y]);
    }
    return chosen;
  }

  // cull pass shared by the main mesh and the background layer:
  // depth ratio 1.4, then edge length <= 4x the median
  function cullDepthEdge(tris, pts, P, W) {
    const prep = [], lens = [];
    for (const t of tris) {
      const Q = t.map(i => P[pts[i][1] * W + pts[i][0]]); if (Q.some(q => !q)) continue;
      const zs = Q.map(q => q[2]), zmin = Math.min(...zs), zmax = Math.max(...zs);
      if (zmin <= 0 || zmax > zmin * 1.4) continue;
      let e = 0; for (let a = 0; a < 3; a++) { const b = (a + 1) % 3; e = Math.max(e, Math.hypot(Q[a][0] - Q[b][0], Q[a][1] - Q[b][1], Q[a][2] - Q[b][2])); }
      prep.push([t, Q, e]); lens.push(e);
    }
    lens.sort((a, b) => a - b);
    const lim = (lens[lens.length >> 1] || 0) * 4;
    return prep.filter(r => r[2] <= lim);
  }

  // Full stage 2: disparity (display resolution) -> mesh.
  // opts.refined: the map came from the full-resolution refinement, so it
  //   is not smoothed (smoothing measurably blurs a refined map).
  // opts.fill: add a background layer behind foreground edges.
  function buildMesh(disp, gray, rgb, W, H, log, opts) {
    log = log || (() => {}); opts = opts || {};
    let d = disp;
    if (!opts.refined) {
      let mx = 0; for (const v of disp) if (v > mx) mx = v;
      d = fixLeftBorder(disp, W, H, Math.trunc(mx + 2));
      d = median3(d, W, H, 2);
      d = bilateral(d, gray, W, H, 4, 2.5, 0.09, 3);
      log('filtered: median 3x3, image-guided bilateral');
    }
    const fx = (W * 0.5) / Math.tan(Math.PI / 6), cx = W / 2, cy = H / 2, B = 0.1;
    const srt = Float64Array.from(d).sort(), dmin = Math.max(0.4, srt[srt.length >> 1] * 0.38);
    const lift = (x, y, v) => { const Z = fx * B / v; return [(x - cx) * Z / fx, (y - cy) * Z / fx, Z]; };
    const P = new Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const v = d[y * W + x]; if (v >= dmin) P[y * W + x] = lift(x, y, v); }
    const target = Math.max(600, Math.min(5000, Math.trunc(W * H / 40)));
    const pts = selectPoints(d, W, H, target);
    let tris = delaunay(pts);
    const nRaw = tris.length;
    const kept = [];
    for (const [t, Q] of cullDepthEdge(tris, pts, P, W)) {
      const u = [0, 1, 2].map(k => Q[1][k] - Q[0][k]), v = [0, 1, 2].map(k => Q[2][k] - Q[0][k]);
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const c = [0, 1, 2].map(k => (Q[0][k] + Q[1][k] + Q[2][k]) / 3);
      const cosang = Math.abs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]) / (Math.hypot(...n) * Math.hypot(...c) || 1);
      if (cosang >= 0.07) kept.push(t);
    }
    tris = kept;
    log('Delaunay: ' + pts.length + ' vertices, ' + nRaw + ' triangles, ' + tris.length + ' kept after culling');
    const verts = [], cols = [], faces = [];
    function emit(points, triangles, Pv, colourAt) {
      const idx = new Int32Array(points.length).fill(-1);
      points.forEach(([x, y], i) => {
        const q = Pv[y * W + x]; if (!q) return;
        idx[i] = verts.length / 3; verts.push(q[0], q[1], q[2]);
        const o = 3 * colourAt(y * W + x); cols.push(rgb[o] * 255 | 0, rgb[o + 1] * 255 | 0, rgb[o + 2] * 255 | 0);
      });
      for (const [a, b, c] of triangles) if (idx[a] >= 0 && idx[b] >= 0 && idx[c] >= 0) faces.push(idx[a], idx[b], idx[c]);
    }
    emit(pts, tris, P, i => i);
    const mainFaces = faces.length / 3;
    if (opts.fill && typeof Refine !== 'undefined') {
      const reg = Refine.backgroundRegion(d, W, H);
      const PL = new Array(W * H);
      for (let i = 0; i < W * H; i++) if (reg.mask[i] && reg.m[i] >= dmin) PL[i] = lift(i % W, Math.floor(i / W), reg.m[i]);
      const s = Math.max(2, Math.trunc(Math.sqrt(W * H / target) + 0.5)), lp = [];
      for (let y = 0; y < H; y += s) for (let x = 0; x < W; x += s) if (reg.mask[y * W + x]) lp.push([x, y]);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x; if (!reg.mask[i]) continue;
        const edge = x === 0 || x === W - 1 || y === 0 || y === H - 1 || !reg.mask[i - 1] || !reg.mask[i + 1] || !reg.mask[i - W] || !reg.mask[i + W];
        if (edge && x % 2 === 0 && y % 2 === 0) lp.push([x, y]);
      }
      if (lp.length >= 3) {
        const lt = delaunay(lp).filter(([a, b, c]) => {
          const mx = Math.trunc((lp[a][0] + lp[b][0] + lp[c][0]) / 3), my = Math.trunc((lp[a][1] + lp[b][1] + lp[c][1]) / 3);
          return reg.mask[my * W + mx];
        });
        const lk = cullDepthEdge(lt, lp, PL, W).map(r => r[0]);
        emit(lp, lk, PL, i => reg.src[i]);
        log('background layer: ' + lk.length + ' triangles continue the background behind foreground edges');
      }
    }
    return { verts: new Float32Array(verts), cols: new Uint8Array(cols), faces: new Int32Array(faces), mainFaces, filtered: d };
  }


  // Dense grid mesh for the WebGL viewer: a vertex every `step` pixels,
  // two triangles per cell, texture coordinates straight from the pixel
  // position so the photograph is mapped onto the surface pixel for pixel.
  // A cell is dropped where disparity jumps (an occlusion edge, not a
  // surface) or falls below the far clamp. With opts.fill, a second grid
  // continues the background behind foreground edges; its texture
  // coordinates point at the background pixel each value was copied from.
  function buildGridMesh(disp, W, H, opts) {
    opts = opts || {};
    const step = opts.step || 2, jump = opts.jump || 1.5, jumpRel = opts.jumpRel || 0.06;
    const fx = (W * 0.5) / Math.tan(Math.PI / 6), cx = W / 2, cy = H / 2, B = 0.1;
    const srt = Float64Array.from(disp).sort(), dmin = Math.max(0.4, srt[srt.length >> 1] * 0.38);
    const xs = [], ys = [];
    for (let x = 0; x < W; x += step) xs.push(x); if (xs[xs.length - 1] !== W - 1) xs.push(W - 1);
    for (let y = 0; y < H; y += step) ys.push(y); if (ys[ys.length - 1] !== H - 1) ys.push(H - 1);
    const nx = xs.length, ny = ys.length;
    const pos = [], uv = [], fill = [], dv = [], idx = [], lines = [];
    function grid(valueAt, uvAt, isFill, inRegion) {
      const base = pos.length / 3, vid = new Int32Array(nx * ny).fill(-1), dd = new Float64Array(nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const x = xs[i], y = ys[j], p = y * W + x;
        if (inRegion && !inRegion(p)) continue;
        const d = valueAt(p); if (!(d >= dmin)) continue;
        // x < d: the right camera never saw this pixel, so nothing was
        // measured there; leave it out rather than draw a smeared guess
        if (!isFill && opts.cropUnseen !== false && x - d < 0) continue;
        const Z = fx * B / d;
        vid[j * nx + i] = pos.length / 3; dd[j * nx + i] = d;
        pos.push((x - cx) * Z / fx, (y - cy) * Z / fx, Z);
        const [u, v] = uvAt(p, x, y); uv.push(u, v); fill.push(isFill ? 1 : 0); dv.push(d);
      }
      let nt = 0;
      const ok = (a, b, c) => {
        if (a < 0 || b < 0 || c < 0) return false;
        const da = dv[a], db = dv[b], dc = dv[c], lo = Math.min(da, db, dc), hi = Math.max(da, db, dc);
        return hi - lo <= Math.max(jump, jumpRel * hi);
      };
      for (let j = 0; j + 1 < ny; j++) for (let i = 0; i + 1 < nx; i++) {
        const a = vid[j * nx + i], b = vid[j * nx + i + 1], c = vid[(j + 1) * nx + i], e = vid[(j + 1) * nx + i + 1];
        if (ok(a, b, c)) { idx.push(a, c, b); nt++; }
        if (ok(b, c, e)) { idx.push(b, c, e); nt++; }
      }
      if (!isFill) {   // wireframe: every 4th grid line
        for (let j = 0; j < ny; j += 4) for (let i = 0; i + 1 < nx; i++) { const a = vid[j * nx + i], b = vid[j * nx + i + 1]; if (a >= 0 && b >= 0 && ok(a, b, b)) lines.push(a, b); }
        for (let i = 0; i < nx; i += 4) for (let j = 0; j + 1 < ny; j++) { const a = vid[j * nx + i], b = vid[(j + 1) * nx + i]; if (a >= 0 && b >= 0 && ok(a, b, b)) lines.push(a, b); }
      }
      return nt;
    }
    const tex = (p, x, y) => [(x + 0.5) / W, (y + 0.5) / H];
    const mainTris = grid(p => disp[p], tex, false, null);
    let fillTris = 0;
    if (opts.fill && typeof Refine !== 'undefined') {
      const reg = Refine.backgroundRegion(disp, W, H);
      fillTris = grid(p => reg.m[p], p => { const s = reg.src[p]; return [((s % W) + 0.5) / W, (Math.floor(s / W) + 0.5) / H]; }, true, p => reg.mask[p]);
    }
    // pivot for orbiting: median depth of the measured surface
    const zs = []; for (let k = 0; k < fill.length; k++) if (!fill[k]) zs.push(pos[3 * k + 2]);
    zs.sort((a, b) => a - b);
    let dlo = Infinity, dhi = -Infinity; for (let k = 0; k < dv.length; k++) { if (!fill[k]) { if (dv[k] < dlo) dlo = dv[k]; if (dv[k] > dhi) dhi = dv[k]; } }
    return { pos: new Float32Array(pos), uv: new Float32Array(uv), fill: new Float32Array(fill), disp: new Float32Array(dv),
             idx: new Uint32Array(idx), lines: new Uint32Array(lines), mainTris, fillTris,
             fx, cx, cy, W, H, pivotZ: zs[zs.length >> 1] || 1, dlo, dhi, verts: pos.length / 3 };
  }

  const api = { buildGridMesh, buildMesh, cullDepthEdge, delaunay, fixLeftBorder, median3, bilateral };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Recon = api;
})(this);
