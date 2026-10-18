"""
Make the images shown in README.md, from your own test run.

    python3 scripts/reproduce_test.py          # writes reproduce_out/
    python3 scripts/make_readme_images.py      # writes docs/images/*.png

Needs numpy, matplotlib and Pillow (harness only; the model has no imports).
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
NET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reproduce_out", "network")
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "reproduce_out", "refined")
INK, INK2 = "#1a1a1a", "#55534e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
os.makedirs(OUT, exist_ok=True)


def pipeline():
    fig, a = plt.subplots(figsize=(6.8, 1.55))
    a.set_xlim(0, 100); a.set_ylim(4, 24); a.axis("off")
    steps = [("Rectified\nstereo pair", "", "#f1f0ec"), ("Siamese\nfeatures", "1/4 res.,\nshared", "#cde2fb"),
             ("Cost volume", "|f$_L$-f$_R$|,\n12 levels", "#b7d3f6"), ("3D filter +\nsoft-argmin", "sub-pixel\ndisparity", "#9ec5f4"),
             ("Guided\nrefine x2", "dilated\nconvs", "#86b6ef"), ("Full-res\nrefinement", "band SGM,\nLR, planes", "#d9d7cf"),
             ("Mesh +\nGaussians", "Delaunay,\nclosed-form $\\Sigma$", "#fde3d6")]
    n = len(steps); w = 11.8; gap = (100 - n * w) / (n - 1)
    for i, (t, sub, col) in enumerate(steps):
        x = i * (w + gap)
        a.add_patch(FancyBboxPatch((x, 7), w, 15, boxstyle="round,pad=0,rounding_size=1.2", fc=col, ec="#bdbab2", lw=0.7))
        a.text(x + w / 2, 18.0, t, ha="center", va="center", fontsize=7, color=INK, fontweight="bold")
        if sub:
            a.text(x + w / 2, 11.0, sub, ha="center", va="center", fontsize=6.2, color=INK2)
        if i < n - 1:
            a.annotate("", xy=(x + w + gap - 0.3, 14.5), xytext=(x + w + 0.3, 14.5),
                       arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def progression():
    labels = ["Synthetic\nonly", "Real\nfine-tune", "Search\n8$\\to$12", "Lower\nLR", "+13 real\nscenes", "+ full-res\nrefinement"]
    vals = [5.21, 3.49, 2.47, 2.45, 2.29, 1.14]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.bar(range(6), vals, color=["#3a7fd0"] * 5 + ["#0d366b"], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, "%.2f" % v, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Test EPE (px), lower is better", fontsize=7.5)
    ax.set_ylim(0, 6); ax.grid(axis="y", color="#e6e4df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#9a978f"); ax.spines["bottom"].set_color("#9a978f"); ax.tick_params(colors=INK2)
    fig.savefig(os.path.join(OUT, "progression.png"), dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def load(d, year, scene):
    c = pickle.load(open(os.path.join(d, "%s_%s.pkl" % (year, scene)), "rb"))
    W, H = c["W"], c["H"]
    shape = (H, W)
    return (np.array(c["pred"], float).reshape(shape), np.array(c["gt"], float).reshape(shape),
            np.array(c["nonocc"], bool).reshape(shape))


def error_maps():
    scenes = [("2003", "teddy", "im2.png", "Teddy (2003)"), ("2005", "Art", "view1.png", "Art (2005)")]
    fig, axes = plt.subplots(len(scenes), 4, figsize=(9.2, 3.9))
    err_cmap = plt.get_cmap("Oranges").copy(); err_cmap.set_bad("#bdbdbd")
    for r, (year, scene, left, title) in enumerate(scenes):
        net, gt, ok = load(NET, year, scene)
        ref, _, _ = load(REF, year, scene)
        img = Image.open(os.path.join(ROOT, "real", year, scene, left)).convert("RGB").resize(gt.shape[::-1])
        lo, hi = np.percentile(gt[ok], [1, 99])
        e_net = np.where(ok, np.minimum(np.abs(net - gt), 4), np.nan)
        e_ref = np.where(ok, np.minimum(np.abs(ref - gt), 4), np.nan)
        panels = [(np.asarray(img), None, "left image"),
                  (np.ma.masked_where(~ok, ref), ("viridis", lo, hi), "refined disparity"),
                  (np.ma.masked_invalid(e_net), (err_cmap, 0, 4), "|error|, network only"),
                  (np.ma.masked_invalid(e_ref), (err_cmap, 0, 4), "|error|, refined")]
        for c, (data, cm, name) in enumerate(panels):
            ax = axes[r, c]
            if cm is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cm[0], vmin=cm[1], vmax=cm[2], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel(title, fontsize=8, color=INK2)
        for c, e in ((2, net), (3, ref)):
            epe = np.abs(e - gt)[ok].mean()
            axes[r, c].text(0.02, 0.03, "EPE %.2f px" % epe, transform=axes[r, c].transAxes, fontsize=7,
                            color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    sm = plt.cm.ScalarMappable(cmap=err_cmap, norm=plt.Normalize(0, 4))
    cb = fig.colorbar(sm, ax=axes[:, 2:].ravel().tolist(), fraction=0.025, pad=0.01)
    cb.set_label("error (px, capped at 4)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.savefig(os.path.join(OUT, "error_maps.png"), dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    pipeline(); progression()
    if os.path.isdir(NET) and os.path.isdir(REF):
        error_maps()
    else:
        print("reproduce_out/ not found: run scripts/reproduce_test.py first for error_maps.png")
    print("images written to", os.path.relpath(OUT, ROOT))