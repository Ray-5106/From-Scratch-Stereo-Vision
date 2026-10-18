"""
Download the Middlebury stereo data used in the paper and lay it out as
eval_real.py expects:

    real/2003/cones/{im2,im6,disp2,disp6}.png
    real/2003/teddy/...
    real/2005/<Scene>/{view1,view5,disp1,disp5}.png      (third size)
    real/2006/<Scene>/...                                 (third size)

    python3 scripts/get_middlebury.py              # evaluation data (~16 MB)
    python3 scripts/get_middlebury.py --training   # also MiddEval3 quarter-size
                                                   # training data (~43 MB)

The datasets belong to their authors (Scharstein et al.); see
https://vision.middlebury.edu/stereo/data/ and cite them if you use them.
HARNESS, NOT THE MODEL: imports are allowed here.
"""
import io, os, shutil, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://vision.middlebury.edu/stereo"
SOURCES = [
    ("2003", BASE + "/data/scenes2003/newdata/cones/cones-png-2.zip"),
    ("2003", BASE + "/data/scenes2003/newdata/teddy/teddy-png-2.zip"),
    ("2005", BASE + "/data/scenes2005/ThirdSize/zip-2views/ALL-2views.zip"),
    ("2006", BASE + "/data/scenes2006/ThirdSize/zip-2views/ALL-2views.zip"),
]
TRAINING = [
    BASE + "/submit3/zip/MiddEval3-data-Q.zip",
    BASE + "/submit3/zip/MiddEval3-GT0-Q.zip",
]
KEEP = {".png"}


def fetch(url):
    print("downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def unpack(data, year, default_scene):
    """Extract PNGs, keeping the last folder name as the scene name (or the
    zip's own name when the archive has no folder)."""
    dest = os.path.join(ROOT, "real", year)
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir() or os.path.splitext(info.filename)[1].lower() not in KEEP:
                continue
            parts = [p for p in info.filename.split("/") if p]
            scene = parts[-2] if len(parts) >= 2 else default_scene
            name = parts[-1]
            os.makedirs(os.path.join(dest, scene), exist_ok=True)
            with z.open(info) as src, open(os.path.join(dest, scene, name), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    print("   %d files -> real/%s" % (n, year))


def main(argv):
    for year, url in SOURCES:
        unpack(fetch(url), year, url.rsplit("/", 1)[1].split("-")[0])
    if "--training" in argv:
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        for url in TRAINING:
            path = os.path.join(ROOT, "data", url.rsplit("/", 1)[1])
            open(path, "wb").write(fetch(url))
            print("   saved", os.path.relpath(path, ROOT))
        print("Then: python3 prep_middeval.py data/MiddEval3-data-Q.zip data/MiddEval3-GT0-Q.zip")
    for year in ("2003", "2005", "2006"):
        d = os.path.join(ROOT, "real", year)
        print("real/%s: %s" % (year, ", ".join(sorted(os.listdir(d)))))


if __name__ == "__main__":
    main(sys.argv)
