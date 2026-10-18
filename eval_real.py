"""
Evaluate on real Middlebury photographs.

    python3 eval_real.py                 # current weights, all splits
    python3 eval_real.py --split test    # only the untouched test set
    python3 eval_real.py --weights other_weights.txt --consistency
    pypy3 eval_real.py --split val --refine --dump preds/refine
    pypy3 eval_real.py --split val --sgm-only --dump preds/sgm
    python3 eval3d.py preds/refine preds/sgm      # 3D metrics, needs numpy

Expects the unzipped datasets laid out as

    real/2006/<Scene>/view1.png view5.png disp1.png disp5.png
    real/2005/<Scene>/...
    real/2003/cones/im2.png im6.png disp2.png disp6.png

TEST HARNESS, NOT THE MODEL. It imports os for directory listing. The
model is loaded as source text; images and ground truth are read through
the from-scratch PNG decoder.

Metrics follow Middlebury practice. Two masks:
  nonocc  -- pixels with known ground truth that are visible in BOTH views,
             found by checking the left ground truth against the right
             ground truth (|dL(x) - dR(x - dL(x))| <= 1). This is the
             headline number; no stereo method can recover the rest.
  all     -- every pixel with known ground truth.
Ground-truth grey level 0 means unknown and is never scored.

THE SPLIT (fixed before any real-data result was looked at):
  2006 scenes, sorted alphabetically; index % 4 == 1 -> validation,
  the rest -> training. All of 2005 and 2003 -> test, never trained on
  and never used to choose anything.
"""

import os
import sys
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "stereonet.py")).read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", "")
SN = {"__name__": "stereonet_lib"}
exec(_src, SN)

LAYOUT = {
    "2006": ("view1.png", "view5.png", "disp1.png", "disp5.png", 3.0),
    "2005": ("view1.png", "view5.png", "disp1.png", "disp5.png", 3.0),
    "2003": ("im2.png", "im6.png", "disp2.png", "disp6.png", 4.0),
}


def scenes(root):
    """All scenes with ground truth, tagged with their split."""
    out = []
    for year in ("2006", "2005", "2003"):
        ydir = os.path.join(root, year)
        if not os.path.isdir(ydir):
            continue
        L, R, DL, DR, scale = LAYOUT[year]
        names = sorted(n for n in os.listdir(ydir)
                       if os.path.isdir(os.path.join(ydir, n)))
        for i, name in enumerate(names):
            d = os.path.join(ydir, name)
            if not os.path.exists(os.path.join(d, DL)):
                continue
            if year == "2006":
                split = "val" if i % 4 == 1 else "train"
            else:
                split = "test"
            out.append({"year": year, "name": name, "dir": d, "split": split,
                        "left": os.path.join(d, L), "right": os.path.join(d, R),
                        "gtl": os.path.join(d, DL), "gtr": os.path.join(d, DR),
                        "scale": scale})
    return out


def load_gt(path, scale):
    g, _, W, H = SN["read_image"](path)
    disp = [0.0] * (W * H)
    known = [False] * (W * H)
    for i in range(W * H):
        lv = g[i] * 255.0
        if lv >= 0.5:
            disp[i] = lv / scale
            known[i] = True
    return disp, known, W, H


def masks(sc):
    dl, kl, W, H = load_gt(sc["gtl"], sc["scale"])
    dr, kr, _, _ = load_gt(sc["gtr"], sc["scale"])
    nonocc = [False] * (W * H)
    for y in range(H):
        row = y * W
        for x in range(W):
            i = row + x
            if not kl[i]:
                continue
            xr = int(x - dl[i] + 0.5)
            if 0 <= xr < W and kr[row + xr] and abs(dl[i] - dr[row + xr]) <= 1.0:
                nonocc[i] = True
    return dl, kl, nonocc, W, H


def score(pred, gt, mask):
    n = 0
    e = 0.0
    b = [0, 0, 0]
    for i in range(len(pred)):
        if not mask[i]:
            continue
        d = abs(pred[i] - gt[i])
        e += d
        if d > 1.0:
            b[0] += 1
        if d > 2.0:
            b[1] += 1
        if d > 4.0:
            b[2] += 1
        n += 1
    if n == 0:
        return None
    return (e / n, 100.0 * b[0] / n, 100.0 * b[1] / n, 100.0 * b[2] / n)


def constant(gt, mask):
    v = [gt[i] for i in range(len(gt)) if mask[i]]
    m = sum(v) / len(v)
    return sum(abs(x - m) for x in v) / len(v)


def evaluate(net, sc, consistency=False, save_dir=None, mode="net",
             dump=None):
    """mode: 'net' (network only), 'refine' (network + SECTION 25), or
    'sgm' (the classical control: the same census + SGM with no network,
    searching the full range 0..max ground-truth disparity + 2, a range
    the control is handed for free)."""
    lg, lrgb, W, H = SN["read_image"](sc["left"])
    rg, _, _, _ = SN["read_image"](sc["right"])
    gt, known, nonocc, Wg, Hg = masks(sc)
    t0 = time.time()
    if mode == "sgm":
        R = int(max(gt[i] for i in range(W * H) if nonocc[i]) + 2) // 2 + 1
        pred = SN["refine_fullres"](lg, rg, [float(R)] * (W * H), W, H, R=R,
                                    P1=12.0, P2=150.0, rx=4, ry=3)
    elif consistency:
        pred, _ = SN["predict_consistent"](net, lg, rg, W, H, verbose=False)
    else:
        pred, _, _, _ = SN["predict_auto"](net, lg, rg, W, H, verbose=False)
        if mode == "refine":
            pred, _ = SN["refine_disparity"](lg, rg, lrgb, pred, W, H)
    dt = time.time() - t0
    if dump:
        os.makedirs(dump, exist_ok=True)
        with open(os.path.join(dump, "%s_%s.pkl" % (sc["year"], sc["name"])),
                  "wb") as f:
            pickle.dump({"scene": sc["name"], "year": sc["year"],
                         "split": sc["split"], "W": W, "H": H, "pred": pred,
                         "gt": gt, "nonocc": nonocc, "known": known}, f)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        SN["write_pgm"](os.path.join(save_dir, "%s_%s.pgm"
                                     % (sc["year"], sc["name"])),
                        pred, W, H, 0.0, max(gt))
    return {"nonocc": score(pred, gt, nonocc), "all": score(pred, gt, known),
            "const": constant(gt, nonocc), "seconds": dt,
            "size": "%dx%d" % (W, H)}


def table(rows, title):
    out = ["", "### " + title, "",
           "| scene | split | EPE | bad1 | bad2 | bad4 | all EPE | constant | s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for sc, r in rows:
        n = r["nonocc"]
        a = r["all"]
        out.append("| %s %s | %s | %.2f | %.1f%% | %.1f%% | %.1f%% | %.2f | "
                   "%.2f | %.0f |" % (sc["year"], sc["name"], sc["split"],
                                      n[0], n[1], n[2], n[3], a[0],
                                      r["const"], r["seconds"]))
    for split in ("train", "val", "test"):
        sub = [r for sc, r in rows if sc["split"] == split]
        if not sub:
            continue
        k = len(sub)
        out.append("| **mean %s (%d)** | | **%.2f** | **%.1f%%** | **%.1f%%** "
                   "| **%.1f%%** | **%.2f** | **%.2f** | |"
                   % (split, k, sum(r["nonocc"][0] for r in sub) / k,
                      sum(r["nonocc"][1] for r in sub) / k,
                      sum(r["nonocc"][2] for r in sub) / k,
                      sum(r["nonocc"][3] for r in sub) / k,
                      sum(r["all"][0] for r in sub) / k,
                      sum(r["const"] for r in sub) / k))
    return "\n".join(out)


def main(argv):
    root = os.path.join(HERE, "real")
    weights = os.path.join(HERE, "stereonet_weights.txt")
    split = None
    cons = "--consistency" in argv
    mode = "refine" if "--refine" in argv else ("sgm" if "--sgm-only" in argv
                                                else "net")
    dump = argv[argv.index("--dump") + 1] if "--dump" in argv else None
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    if "--weights" in argv:
        weights = argv[argv.index("--weights") + 1]
    if "--split" in argv:
        split = argv[argv.index("--split") + 1]
    net = SN["load_model"](weights)
    rows = []
    for sc in scenes(root):
        if split and sc["split"] != split:
            continue
        r = evaluate(net, sc, cons, os.path.join(HERE, "real_out"), mode,
                     dump)
        rows.append((sc, r))
        n = r["nonocc"]
        print("%s %-11s %-5s EPE %5.2f  bad2 %5.1f%%  constant %5.2f  (%.0fs)"
              % (sc["year"], sc["name"], sc["split"], n[0], n[2], r["const"],
                 r["seconds"]), flush=True)
    title = "%s%s" % (os.path.basename(weights),
                      " + left-right consistency" if cons else "")
    if mode == "refine":
        title += " + full-resolution refinement and planes"
    elif mode == "sgm":
        title = "classical census + SGM alone (no network), range from ground truth"
    txt = table(rows, title)
    print(txt)
    with open(os.path.join(HERE, "real_results.md"), "a") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main(sys.argv)
