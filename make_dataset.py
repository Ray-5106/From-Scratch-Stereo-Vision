"""
Render a dataset of physically ray-traced stereo pairs with exact ground truth.

    python3 make_dataset.py train 200 128 96 datasets/rendered_train
    python3 make_dataset.py eval   24 320 240 datasets/rendered_eval

Each scene writes three files the from-scratch pipeline can read without
any imports:

    NNNN_left.png    NNNN_right.png    NNNN_disp.pgm   (16-bit)

TEST FIXTURE, NOT THE MODEL. This file imports numpy and PIL. Its only
job is to manufacture input; stereonet.py never touches it and still has
zero imports.

Why this exists. The generators inside stereonet.py build the right image
by warping the left through a known disparity. That is enough to teach
matching, but it cannot produce real perspective, real occlusion or real
shading, because there is no actual 3D scene behind it. Here there is:
two pinhole cameras at a horizontal baseline observing spheres, boxes, a
ground plane and a back wall. Both views are rendered independently from
the geometry, which is exactly what a stereo rig does and what a warp can
never simulate.

The baseline is chosen per scene so the maximum disparity lands in the
network's search range: depth is rendered first, then B = d_target * Zmin / f.
"""

import os
import sys
import numpy as np
from PIL import Image


def noise2d(rng, shape, octaves=5, persistence=0.55):
    acc = np.zeros(shape)
    amp, tot, n = 1.0, 0.0, 4
    for _ in range(octaves):
        g = rng.random((n + 1, n + 1))
        im = np.array(Image.fromarray((g * 255).astype(np.uint8))
                      .resize((shape[1], shape[0]), Image.BICUBIC)) / 255.0
        acc += amp * im
        tot += amp
        amp *= persistence
        n *= 2
    return acc / tot


class Scene:
    def __init__(self, rng, W, H):
        self.rng = rng
        self.W, self.H = W, H
        self.fx = (W * 0.5) / np.tan(np.deg2rad(rng.uniform(26, 34)))
        self.cx, self.cy = W / 2.0, H / 2.0
        self.gy = -0.85
        # Wall kept close and objects brought forward. An earlier version
        # put the wall at 3-5 m with objects at 1.2-3 m, which produced
        # crops whose disparity was nearly constant -- predicting the mean
        # beat the network, and training on them would have taught
        # smoothness rather than matching.
        self.wallz = rng.uniform(2.6, 4.0)

        self.spheres = []
        for _ in range(rng.integers(3, 7)):
            r = rng.uniform(0.20, 0.50)
            self.spheres.append((
                (rng.uniform(-0.95, 0.95), self.gy + r, rng.uniform(0.85, 2.30)),
                r,
                tuple(rng.uniform(0.15, 0.97, 3))))
        self.boxes = []
        for _ in range(rng.integers(2, 5)):
            s = rng.uniform(0.14, 0.48, 3)
            self.boxes.append((
                (rng.uniform(-1.0, 1.0), self.gy + s[1], rng.uniform(0.80, 2.30)),
                tuple(s), tuple(rng.uniform(0.15, 0.97, 3))))

        self.tg = noise2d(rng, (512, 512), octaves=rng.integers(3, 6))
        self.tw = noise2d(rng, (512, 512), octaves=rng.integers(3, 6))
        self.ts = noise2d(rng, (256, 256), octaves=rng.integers(4, 7))
        self.gscale = rng.uniform(0.8, 2.6)
        self.wscale = rng.uniform(0.25, 0.9)
        self.gtex = rng.uniform(0.10, 0.45)     # texture contrast varies:
        self.wtex = rng.uniform(0.08, 0.40)     # some surfaces near-blank
        self.stex = rng.uniform(0.15, 0.55)
        L = rng.normal(size=3) * np.array([1.0, 0.6, 0.7]) + np.array([0, 0.9, -0.5])
        self.light = L / np.linalg.norm(L)
        self.ambient = rng.uniform(0.22, 0.40)

    def render(self, eye_x):
        W, H = self.W, self.H
        ys, xs = np.mgrid[0:H, 0:W]
        dx = (xs - self.cx) / self.fx
        dy = -(ys - self.cy) / self.fx
        dz = np.ones_like(dx)
        n = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / n, dy / n, dz / n
        ox = np.full_like(dx, eye_x)
        oy = np.zeros_like(dx)
        oz = np.zeros_like(dx)

        best = np.full((H, W), 1e9)
        col = np.zeros((H, W, 3))
        nrm = np.zeros((H, W, 3))

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(dy < -1e-6, (self.gy - oy) / dy, 1e9)
        t = np.where((t > 0) & np.isfinite(t), t, 1e9)
        px, pz = ox + t * dx, oz + t * dz
        hit = (t < best) & (pz < self.wallz) & (pz > 0)
        u = (((px * self.gscale) % 1.0) * 511).astype(int)
        v = (((pz * self.gscale) % 1.0) * 511).astype(int)
        g = self.tg[v, u]
        base = np.stack([0.40 + self.gtex * g, 0.38 + self.gtex * g,
                         0.35 + self.gtex * g], -1)
        col = np.where(hit[..., None], base, col)
        nrm = np.where(hit[..., None], np.array([0, 1.0, 0]), nrm)
        best = np.where(hit, t, best)

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(dz > 1e-6, (self.wallz - oz) / dz, 1e9)
        t = np.where((t > 0) & np.isfinite(t), t, 1e9)
        px, py = ox + t * dx, oy + t * dy
        hit = (t < best) & (py > self.gy)
        u = ((((px + 3) * self.wscale) % 1.0) * 511).astype(int)
        v = ((((py + 2) * self.wscale) % 1.0) * 511).astype(int)
        g = self.tw[v, u]
        base = np.stack([0.55 + self.wtex * g, 0.54 + self.wtex * g,
                         0.52 + self.wtex * g], -1)
        col = np.where(hit[..., None], base, col)
        nrm = np.where(hit[..., None], np.array([0, 0, -1.0]), nrm)
        best = np.where(hit, t, best)

        for (c, r, alb) in self.spheres:
            cxx, cyy, czz = c
            Lx, Ly, Lz = cxx - ox, cyy - oy, czz - oz
            tca = Lx * dx + Ly * dy + Lz * dz
            d2 = Lx * Lx + Ly * Ly + Lz * Lz - tca * tca
            ok = d2 <= r * r
            t = tca - np.sqrt(np.maximum(r * r - d2, 0))
            t = np.where(ok & (t > 0), t, 1e9)
            hit = t < best
            pxx, pyy, pzz = ox + t * dx, oy + t * dy, oz + t * dz
            nx, ny, nz = (pxx - cxx) / r, (pyy - cyy) / r, (pzz - czz) / r
            uu = (((np.arctan2(nz, nx) / (2 * np.pi) + 0.5) % 1.0) * 255).astype(int)
            vv = (((np.arccos(np.clip(ny, -1, 1)) / np.pi) % 1.0) * 255).astype(int)
            g = self.ts[vv, uu]
            k = 1.0 - self.stex + 2 * self.stex * g
            base = np.stack([alb[0] * k, alb[1] * k, alb[2] * k], -1)
            col = np.where(hit[..., None], base, col)
            nrm = np.where(hit[..., None], np.stack([nx, ny, nz], -1), nrm)
            best = np.where(hit, t, best)

        D = np.stack([dx, dy, dz], -1)
        O = np.stack([ox, oy, oz], -1)
        for (c, s, alb) in self.boxes:
            lo = np.array(c) - np.array(s)
            hi = np.array(c) + np.array(s)
            inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
            t0, t1 = (lo - O) * inv, (hi - O) * inv
            tmin = np.minimum(t0, t1).max(-1)
            tmax = np.maximum(t0, t1).min(-1)
            t = np.where(tmax >= np.maximum(tmin, 0), tmin, 1e9)
            t = np.where(t > 0, t, 1e9)
            hit = t < best
            pxx, pyy, pzz = ox + t * dx, oy + t * dy, oz + t * dz
            nx = np.where(np.abs(pxx - lo[0]) < 1e-3, -1.0,
                          np.where(np.abs(pxx - hi[0]) < 1e-3, 1.0, 0.0))
            ny = np.where(np.abs(pyy - lo[1]) < 1e-3, -1.0,
                          np.where(np.abs(pyy - hi[1]) < 1e-3, 1.0, 0.0))
            nz = np.where(np.abs(pzz - lo[2]) < 1e-3, -1.0,
                          np.where(np.abs(pzz - hi[2]) < 1e-3, 1.0, 0.0))
            u = (((pxx * 2.0) % 1.0) * 511).astype(int)
            v = (((pyy * 2.0) % 1.0) * 511).astype(int)
            g = self.tw[v, u]
            k = 0.82 + 0.36 * g
            base = np.stack([alb[0] * k, alb[1] * k, alb[2] * k], -1)
            col = np.where(hit[..., None], base, col)
            nrm = np.where(hit[..., None], np.stack([nx, ny, nz], -1), nrm)
            best = np.where(hit, t, best)

        nl = np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm = nrm / np.where(nl < 1e-9, 1, nl)
        lam = np.clip((nrm * self.light).sum(-1), 0, 1)
        img = np.clip(col * (self.ambient + (1 - self.ambient) * lam)[..., None], 0, 1)
        z = np.where(best > 1e8, np.nan, best * dz)
        return img, z


def write_pgm16(path, disp, scale=64.0):
    """16-bit PGM so sub-pixel ground truth survives. Value 0 = unknown,
    everything else is disparity * scale."""
    H, W = disp.shape
    v = np.nan_to_num(disp, nan=0.0) * scale
    v = np.clip(np.round(v), 1, 65535).astype(np.uint16)
    v[~np.isfinite(disp)] = 0
    with open(path, 'wb') as f:
        f.write(b'P5\n%d %d\n65535\n' % (W, H))
        f.write(v.astype('>u2').tobytes())


def make(seed, W, H, target_disp):
    rng = np.random.default_rng(seed)
    sc = Scene(rng, W, H)
    imgL, zL = sc.render(0.0)
    zmin = np.nanmin(zL)
    base = target_disp * zmin / sc.fx
    imgR, _ = sc.render(base)
    disp = sc.fx * base / zL
    return imgL, imgR, disp, sc.fx, base


def main(argv):
    split = argv[1] if len(argv) > 1 else 'train'
    count = int(argv[2]) if len(argv) > 2 else 50
    W = int(argv[3]) if len(argv) > 3 else 128
    H = int(argv[4]) if len(argv) > 4 else 96
    out = argv[5] if len(argv) > 5 else os.path.join('datasets', 'rendered_' + split)
    dlo = float(argv[6]) if len(argv) > 6 else 12.0
    dhi = float(argv[7]) if len(argv) > 7 else 16.0
    os.makedirs(out, exist_ok=True)
    base_seed = 10_000 if split == 'train' else 90_000
    lo, hi = [], []
    for i in range(count):
        td = np.random.default_rng(base_seed + i).uniform(dlo, dhi)
        L, R, D, fx, b = make(base_seed + i, W, H, td)
        Image.fromarray((L * 255).astype(np.uint8)).save(
            os.path.join(out, '%04d_left.png' % i))
        Image.fromarray((R * 255).astype(np.uint8)).save(
            os.path.join(out, '%04d_right.png' % i))
        write_pgm16(os.path.join(out, '%04d_disp.pgm' % i), D)
        lo.append(np.nanmin(D))
        hi.append(np.nanmax(D))
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, count), flush=True)
    # An index file, because the pure-Python trainer cannot list a
    # directory: os.listdir is an import. It reads this with open().
    with open(os.path.join(out, 'index.txt'), 'w') as f:
        f.write('%d\n' % count)
        for i in range(count):
            f.write('%04d\n' % i)

    print('%s: %d scenes at %dx%d in %s' % (split, count, W, H, out))
    print('  disparity min %.1f  max %.1f  (mean max %.1f)'
          % (min(lo), max(hi), float(np.mean(hi))))


if __name__ == '__main__':
    main(sys.argv)
