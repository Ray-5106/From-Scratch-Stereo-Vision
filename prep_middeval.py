"""
Prepare MiddEval3 quarter-resolution training scenes for the PyPy trainer.

    python3 prep_middeval.py <MiddEval3-data-Q.zip> <MiddEval3-GT0-Q.zip>

Harness, not model (numpy + PIL for resampling). Writes
real_train_samples2.pkl, which train_parallel.py picks up alongside the
2006 training samples.

EXCLUDED: Teddy and ArtL. MiddEval3 reuses them from the 2003 and 2005
datasets, and those are our TEST scenes. Training on them would turn the
test set into a training set and every test number into a lie. The
exclusion is enforced by name and then asserted.

Ground truth: disp0GT.pfm (float32, rows stored bottom to top, inf where
unknown) and mask0nocc.png (255 = visible in both views, 128 = occluded,
0 = no ground truth). Only the left view has ground truth, so these
scenes are not mirrored.
"""
import io, os, sys, pickle, zipfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {"Teddy", "ArtL"}
PEAKS = (22.0, 30.0, 38.0)


def read_pfm(data):
    f = io.BytesIO(data)
    kind = f.readline().strip()
    if kind not in (b"Pf", b"PF"):
        raise ValueError("not a PFM file")
    W, H = map(int, f.readline().split())
    scale = float(f.readline())
    dt = "<f4" if scale < 0 else ">f4"
    ch = 1 if kind == b"Pf" else 3
    a = np.frombuffer(f.read(W * H * ch * 4), dtype=dt).reshape(H, W, ch)[..., 0]
    return np.flipud(a).astype(np.float32)         # PFM rows run bottom-up


def to_sample(L, R, g, v):
    gray = lambda im: 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
    H, W = g.shape
    return {"left": gray(L).ravel().tolist(), "right": gray(R).ravel().tolist(),
            "disp": g.ravel().astype(float).tolist(), "valid": v.ravel().tolist(),
            "left_rgb": None, "right_rgb": None, "W": W, "H": H, "max_disp": 40}


def main(data_zip, gt_zip):
    zd, zg = zipfile.ZipFile(data_zip), zipfile.ZipFile(gt_zip)
    names = sorted({n.split("/")[-2] for n in zg.namelist()
                    if n.endswith("disp0GT.pfm") and "/training" in n})
    used = [n for n in names if n not in EXCLUDE]
    assert not (set(used) & EXCLUDE)
    print("MiddEval3 training scenes: %d; excluded (overlap with our test set): %s"
          % (len(names), ", ".join(sorted(EXCLUDE & set(names)))))
    find = lambda z, scene, fn: next(n for n in z.namelist()
                                     if n.endswith("/%s/%s" % (scene, fn)))
    out = []
    for sc in used:
        L = np.asarray(Image.open(io.BytesIO(zd.read(find(zd, sc, "im0.png")))).convert("RGB"), np.float32) / 255
        R = np.asarray(Image.open(io.BytesIO(zd.read(find(zd, sc, "im1.png")))).convert("RGB"), np.float32) / 255
        g = read_pfm(zg.read(find(zg, sc, "disp0GT.pfm")))
        m = np.asarray(Image.open(io.BytesIO(zg.read(find(zg, sc, "mask0nocc.png")))))
        ok = np.isfinite(g) & (m == 255)
        g = np.where(np.isfinite(g), g, 0.0)
        H, W = g.shape
        peak = float(np.percentile(g[ok], 99))
        for target in PEAKS:
            s = target / peak
            W2, H2 = max(96, int(W * s + 0.5)), max(64, int(H * s + 0.5))
            sx = W2 / W
            dn = lambda im: np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((W2, H2), Image.BOX), np.float32) / 255
            g2 = np.asarray(Image.fromarray(g, "F").resize((W2, H2), Image.NEAREST)) * sx
            v2 = np.asarray(Image.fromarray(ok.astype(np.uint8) * 255).resize((W2, H2), Image.NEAREST)) > 127
            out.append(to_sample(dn(L), dn(R), g2, v2 & (g2 > 0)))
        print("  %-12s %dx%d  peak disparity %.0f px" % (sc, W, H, peak))
    with open(os.path.join(HERE, "real_train_samples2.pkl"), "wb") as f:
        pickle.dump(out, f, protocol=2)
    print("%d samples from %d scenes -> real_train_samples2.pkl" % (len(out), len(used)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
