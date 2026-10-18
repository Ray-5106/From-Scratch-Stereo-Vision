"""
Fine-tune the network on real Middlebury photographs.

    python3 finetune_real.py [iterations]

TRAINING HARNESS, NOT THE MODEL. It uses numpy and PIL to *prepare* data
(area-downscaling images, nearest-sampling ground truth, mirroring), then
hands plain Python lists to the model's own pure-Python training step.
Every gradient is still computed by stereonet.py.

Two problems this fixes, both measured on the validation set before any
change was made:

  1. RANGE. Real scenes at working scale reach ~26 px of disparity; the
     model had never trained above 16, and on validation it
     underestimated the nearest objects by 7.2 px on average. Training
     scenes are therefore prepared at three scales, putting their peak
     disparity at 16, 21 and 26 px, and the synthetic generators are
     widened to 24.
  2. DOMAIN. Far, low-texture backgrounds were overestimated by 3.5 px:
     real texture and lighting the model had never seen.

Discipline: trains only on the 16 'train' scenes of the fixed split.
Checkpoints are chosen by validation EPE. The test set (2005 + 2003) is
never read here.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "stereonet.py")).read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", "")
SN = {"__name__": "stereonet_lib"}
exec(_src, SN)
_ev = open(os.path.join(HERE, "eval_real.py")).read().replace(
    'if __name__ == "__main__":\n    main(sys.argv)', "")
EV = {"__name__": "eval_lib", "__file__": os.path.join(HERE, "eval_real.py")}
exec(_ev, EV)

ROOT = os.path.join(HERE, "real")
PEAKS = (16.0, 21.0, 26.0)


def load_np(sc):
    L = np.asarray(Image.open(sc["left"]).convert("RGB"), dtype=np.float32) / 255
    R = np.asarray(Image.open(sc["right"]).convert("RGB"), dtype=np.float32) / 255
    gl = np.asarray(Image.open(sc["gtl"]), dtype=np.float32) / sc["scale"]
    gr = np.asarray(Image.open(sc["gtr"]), dtype=np.float32) / sc["scale"]
    return L, R, gl, gr


def nonocc(ga, gb, sign):
    """Pixels of view A with ground truth that are visible in view B.
    sign=-1 for left->right (x - d), +1 for right->left (x + d)."""
    H, W = ga.shape
    out = np.zeros((H, W), bool)
    xs = np.arange(W)[None, :].repeat(H, 0)
    ys = np.arange(H)[:, None].repeat(W, 1)
    xt = np.rint(xs + sign * ga).astype(int)
    ok = (ga > 0) & (xt >= 0) & (xt < W)
    xt = np.clip(xt, 0, W - 1)
    gbv = gb[ys, xt]
    out = ok & (gbv > 0) & (np.abs(ga - gbv) <= 1.0)
    return out


def to_sample(L, R, g, v):
    H, W = g.shape
    gray = lambda im: (0.299 * im[..., 0] + 0.587 * im[..., 1] +
                       0.114 * im[..., 2])
    return {"left": gray(L).ravel().tolist(), "right": gray(R).ravel().tolist(),
            "disp": g.ravel().astype(float).tolist(),
            "valid": v.ravel().tolist(), "left_rgb": L.ravel().tolist(),
            "right_rgb": None, "W": W, "H": H, "max_disp": 26}


def prepare(sc):
    """One real scene -> several cached training samples: three scales,
    each in its natural and its mirrored form. Mirroring swaps the views
    and uses the *right* ground truth, which the 2-view zips provide."""
    L, R, gl, gr = load_np(sc)
    H, W = gl.shape
    vl = nonocc(gl, gr, -1)
    vr = nonocc(gr, gl, +1)
    peak = float(np.percentile(gl[gl > 0], 99))
    out = []
    for target in PEAKS:
        s = target / peak
        W2, H2 = max(96, int(W * s + 0.5)), max(64, int(H * s + 0.5))
        sx = W2 / W
        dn = lambda im: np.asarray(Image.fromarray((im * 255).astype(np.uint8))
                                   .resize((W2, H2), Image.BOX),
                                   dtype=np.float32) / 255
        nn = lambda a: np.asarray(Image.fromarray(a.astype(np.float32), 'F')
                                  .resize((W2, H2), Image.NEAREST))
        nb = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8) * 255)
                                  .resize((W2, H2), Image.NEAREST)) > 127
        L2, R2 = dn(L), dn(R)
        gl2, gr2 = nn(gl) * sx, nn(gr) * sx
        vl2, vr2 = nb(vl) & (gl2 > 0), nb(vr) & (gr2 > 0)
        out.append(to_sample(L2, R2, gl2, vl2))
        out.append(to_sample(R2[:, ::-1].copy(), L2[:, ::-1].copy(),
                             gr2[:, ::-1].copy(), vr2[:, ::-1].copy()))
    return out


def jitter(sample, rng):
    """Independent photometric gain and bias per view. Real pairs never
    match exactly in exposure; a model trained on perfectly matched
    synthetic pairs would treat any brightness difference as mismatch."""
    s = dict(sample)
    for key in ("left", "right"):
        g = rng.range(0.88, 1.12)
        b = rng.range(-0.05, 0.05)
        s[key] = [min(1.0, max(0.0, v * g + b)) for v in sample[key]]
    return s


class Mixed:
    def __init__(self, real, rendered_dir):
        self.real = real
        self.rend = SN["RenderedSampler"](rendered_dir, 0.0, 0.0, cache=40)

    def __call__(self, rng, W, H, md):
        u = rng.uniform()
        if u < 0.55:
            s = self.real[rng.randint(0, len(self.real) - 1)]
            return jitter(SN["crop_sample"](s, W, H, rng), rng)
        if u < 0.80:
            return SN["crop_sample"](self.rend._scene(rng), W, H, rng)
        if u < 0.92:
            return SN["make_stereogram"](rng, W, H, max_disp=24,
                                         n_objects=rng.randint(2, 4))
        return SN["make_textured_scene"](rng, W, H, max_disp=24,
                                         n_objects=rng.randint(2, 4))


def validate(net, vcache):
    eps = []
    for (lg, rg, W, H, gt, non) in vcache:
        pred, _, _, _ = SN["predict_auto"](net, lg, rg, W, H, verbose=False)
        eps.append(EV["score"](pred, gt, non)[0])
    return sum(eps) / len(eps), eps


def main(argv):
    iters = int(argv[1]) if len(argv) > 1 else 1600
    allsc = EV["scenes"](ROOT)
    train = [s for s in allsc if s["split"] == "train"]
    val = [s for s in allsc if s["split"] == "val"]
    assert not any(s["split"] == "test" for s in train + val)
    print("train %d scenes, val %d scenes; test set not loaded"
          % (len(train), len(val)), flush=True)

    t0 = time.time()
    real = []
    for sc in train:
        real += prepare(sc)
    print("prepared %d real training samples (3 scales x 2 mirrors) in %.0fs"
          % (len(real), time.time() - t0), flush=True)

    vcache = []
    for sc in val:
        lg, _, W, H = SN["read_image"](sc["left"])
        rg, _, _, _ = SN["read_image"](sc["right"])
        gt, _, non, _, _ = EV["masks"](sc)
        vcache.append((lg, rg, W, H, gt, non))

    start = os.path.join(HERE, "stereonet_weights.txt")
    net = SN["load_model"](start)
    v0, _ = validate(net, vcache)
    print("start: val EPE %.3f" % v0, flush=True)

    best = v0
    best_path = os.path.join(HERE, "stereonet_real.txt")
    SN["save_weights"](net, best_path)
    opt = SN["Adam"](net.all_params(), lr=8.0e-4)
    sampler = Mixed(real, "/tmp/ds/train")
    rng = SN["Rng"](424242)
    run = 0.0
    log = open("/tmp/ft_log.txt", "a")
    for it in range(1, iters + 1):
        if it == int(iters * 0.6):
            opt.lr *= 0.35
        if it == int(iters * 0.85):
            opt.lr *= 0.35
        loss, _, _ = SN["train_step"](net, opt, sampler(rng, 64, 64, 24))
        run += loss
        if it % 50 == 0:
            line = "iter %4d  loss %.4f  (%.0f min)" % (it, run / 50,
                                                       (time.time() - t0) / 60)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            run = 0.0
        if it % 200 == 0 or it == iters:
            v, per = validate(net, vcache)
            tag = ""
            if v < best:
                best = v
                SN["save_weights"](net, best_path)
                tag = "  <- best, saved"
            line = ("   VAL EPE %.3f  [%s]%s"
                    % (v, " ".join("%.2f" % e for e in per), tag))
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    print("best val EPE %.3f (start %.3f) -> %s" % (best, v0, best_path),
          flush=True)


if __name__ == "__main__":
    main(sys.argv)
