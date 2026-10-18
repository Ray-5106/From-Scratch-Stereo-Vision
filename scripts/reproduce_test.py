"""
Reproduce the test-set numbers reported in the paper (Tables 5 and 7).

    python3 scripts/reproduce_test.py            # uses pypy3 if installed
    python3 scripts/reproduce_test.py --fast     # skip the SGM control

Needs the Middlebury data in ./real (run scripts/get_middlebury.py first),
plus numpy and scipy for the 3D metrics. Runs three configurations on the
8 test scenes, scores them, and compares every per-scene number with
scripts/expected_results.json. Prints PASS when all numbers match to
the stated tolerance.

HARNESS, NOT THE MODEL: imports are allowed here. All disparity
computation happens in stereonet.py, which has none.
"""
import json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOL = 0.005          # metrics are printed to 2 decimals in the paper

sys.path.insert(0, ROOT)
from eval3d import metrics   # noqa: E402

CONFIGS = [("network", []), ("refined", ["--refine"]), ("sgm_control", ["--sgm-only"])]


def run(name, flags, interp, out):
    d = os.path.join(out, name)
    shutil.rmtree(d, ignore_errors=True)
    cmd = [interp, os.path.join(ROOT, "eval_real.py"), "--split", "test", "--dump", d] + flags
    t = time.time()
    print("running %-12s %s" % (name, " ".join(cmd[1:])), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    print("   done in %.0f s" % (time.time() - t), flush=True)
    return d


def score(d):
    import pickle, numpy as np
    res = {}
    for fn in sorted(os.listdir(d)):
        c = pickle.load(open(os.path.join(d, fn), "rb"))
        m = metrics(c["pred"], c["gt"], c["nonocc"], c["W"], c["H"])
        known = np.array(c["known"], bool)
        m["all"] = float(np.abs(np.array(c["pred"]) - np.array(c["gt"]))[known].mean())
        res[c["scene"]] = {k: float(v) for k, v in m.items()}
    return res


def main(argv):
    interp = shutil.which("pypy3") or sys.executable
    print("interpreter for the model:", interp)
    if not os.path.isdir(os.path.join(ROOT, "real", "2005")):
        sys.exit("Middlebury data not found in ./real - run scripts/get_middlebury.py first.")
    expected = json.load(open(os.path.join(HERE, "expected_results.json")))["configs"]
    out = os.path.join(ROOT, "reproduce_out")
    os.makedirs(out, exist_ok=True)
    configs = CONFIGS[:2] if "--fast" in argv else CONFIGS
    ok = True
    for name, flags in configs:
        got = score(run(name, flags, interp, out))
        exp = expected[name]
        keys = ["epe", "bad2", "all", "absrel", "normal", "chamfer"]
        print("\n%-12s %s" % (name, "  ".join("%8s" % k for k in keys)))
        for scene in sorted(got):
            bad = [k for k in keys if abs(got[scene][k] - exp[scene][k]) > TOL]
            ok &= not bad
            print("  %-10s %s %s" % (scene, "  ".join("%8.2f" % got[scene][k] for k in keys),
                                      "" if not bad else "MISMATCH " + ",".join(bad)))
        mean = {k: sum(v[k] for v in got.values()) / len(got) for k in keys}
        print("  %-10s %s" % ("MEAN", "  ".join("%8.2f" % mean[k] for k in keys)))
    print("\n" + ("PASS: every number matches the paper." if ok else
                  "FAIL: some numbers differ; see MISMATCH lines above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
