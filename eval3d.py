"""3D accuracy metrics for predictions dumped by eval_real.py --dump.
Harness, not model: numpy and scipy are allowed here.

    python3 eval3d.py preds/net preds/refine

Geometry uses a nominal camera: f = W px, principal point at the centre,
baseline 1, and the stored disparity taken as the full disparity (the
Middlebury 2005/2006 offset dmin is not in the files we have). Ignoring a
positive offset makes relative depth error an upper bound."""
import numpy as np
from scipy.spatial import cKDTree

def cloud(d, W, H, mask):
    d = np.asarray(d, float).reshape(H, W)
    f = float(W)
    y, x = np.mgrid[0:H, 0:W]
    Z = f / np.maximum(d, 0.5)
    P = np.stack([(x - W/2) * Z / f, (y - H/2) * Z / f, Z], -1)
    return P, mask.reshape(H, W)

def normals(P, s=4):
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, s:-s] = P[:, 2*s:] - P[:, :-2*s]
    dy[s:-s] = P[2*s:] - P[:-2*s]
    n = np.cross(dx, dy)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

def metrics(pred, gt, nonocc, W, H):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float); m = np.asarray(nonocc, bool)
    e = np.abs(pred - gt)[m]
    out = dict(epe=e.mean(), bad1=100*(e > 1).mean(), bad2=100*(e > 2).mean(), bad4=100*(e > 4).mean())
    out['absrel'] = 100 * np.abs(gt[m] / np.maximum(pred[m], 0.5) - 1).mean()
    Pp, M = cloud(pred, W, H, m); Pg, _ = cloud(gt, W, H, m)
    s = 4
    M2 = M.copy()
    M2[:, :s] = M2[:, -s:] = False; M2[:s] = M2[-s:] = False
    # all four neighbours used by the normal must be valid too
    Mn = M2 & np.roll(M, s, 1) & np.roll(M, -s, 1) & np.roll(M, s, 0) & np.roll(M, -s, 0)
    npred, ngt = normals(Pp, s), normals(Pg, s)
    cos = np.clip((npred * ngt).sum(-1), -1, 1)
    out['normal'] = np.degrees(np.arccos(np.abs(cos)))[Mn].mean()
    a = Pp[M][::2]; b = Pg[M][::2]
    med = np.median(b[:, 2])
    ta, tb = cKDTree(a), cKDTree(b)
    out['chamfer'] = 100 * 0.5 * (tb.query(a)[0].mean() + ta.query(b)[0].mean()) / med
    return out

KEYS = ['epe', 'bad1', 'bad2', 'bad4', 'absrel', 'normal', 'chamfer']
def fmt(r): return '  '.join('%s %.2f' % (k, r[k]) for k in KEYS)


if __name__ == '__main__':
    import sys, glob, os, pickle
    rows = []
    for d in sys.argv[1:]:
        res = {}
        for fn in sorted(glob.glob(os.path.join(d, '*.pkl'))):
            c = pickle.load(open(fn, 'rb'))
            res.setdefault(c['split'], []).append(
                (c['scene'], metrics(c['pred'], c['gt'], c['nonocc'], c['W'], c['H'])))
        for split, lst in res.items():
            print('\n%s  [%s, %d scenes]' % (d, split, len(lst)))
            for name, r in lst:
                print('  %-10s ' % name + fmt(r))
            mean = {k: sum(r[k] for _, r in lst) / len(lst) for k in KEYS}
            print('  %-10s ' % 'MEAN' + fmt(mean))
