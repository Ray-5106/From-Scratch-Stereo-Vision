"""
Batch training across every core, under PyPy.

    pypy3 train_parallel.py  <steps>  <start_weights>  <out_weights>

TRAINING HARNESS, NOT THE MODEL. Uses multiprocessing and pickle, both
standard library, to run one worker per core. Each worker holds its own
copy of the network, receives the current weights, computes gradients on
its own sample with stereonet.train_grads, and sends them back; the
parent averages them and takes a single Adam step. That is batch
training with no change to the model code, and batch size 2 also damps
the checkpoint-to-checkpoint swings that batch size 1 produced on real
data.

Every gradient is still computed by the hand-written backward passes in
stereonet.py. PyPy runs the identical source roughly nine times faster
than CPython; the test suite passes unchanged under both.

Only 'train' scenes are used for training, 'val' scenes for choosing the
checkpoint. The test split is never loaded.
"""
import os, sys, time, pickle, multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "stereonet.py")).read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", "")
EVSRC = open(os.path.join(HERE, "eval_real.py")).read().replace(
    'if __name__ == "__main__":\n    main(sys.argv)', "")

D_NEW = 12
TARGET_PEAK = 34.0
CROP_W, CROP_H = 96, 64

G = {}


def build_net(SN, weights):
    cfg = SN["read_checkpoint_config"](weights) or {}
    net = SN["StereoNet"](SN["Rng"](1), feat_C=cfg.get("feat_C", 12),
                          cost_C=cfg.get("cost_C", 6), D=D_NEW, max_disp=48,
                          refine_C=cfg.get("refine_C", 8),
                          refine_dilations=cfg.get("refine_dil", (1, 2, 4)))
    SN["load_weights"](net, weights)      # 3D conv weights are D-independent
    net.target_peak = int(TARGET_PEAK)    # the operating point travels with it
    return net


def init_worker(weights):
    SN = {"__name__": "sn"}
    exec(SRC, SN)
    G["SN"] = SN
    G["net"] = build_net(SN, weights)
    with open(os.path.join(HERE, "real_train_samples.pkl"), "rb") as f:
        G["real"] = pickle.load(f)
    extra = os.path.join(HERE, "real_train_samples2.pkl")
    if os.path.exists(extra):                 # MiddEval3, test overlaps removed
        with open(extra, "rb") as f:
            G["real"] += pickle.load(f)
    G["rend"] = SN["RenderedSampler"](os.environ.get("RENDERED_DIR", os.path.join(HERE, "datasets", "wide")), 0.0, 0.0, cache=50)
    EV = {"__name__": "ev", "__file__": os.path.join(HERE, "eval_real.py")}
    exec(EVSRC, EV)
    G["EV"] = EV
    G["val"] = None


def draw(rng):
    SN = G["SN"]
    u = rng.uniform()
    if u < 0.55:
        s = G["real"][rng.randint(0, len(G["real"]) - 1)]
        c = SN["crop_sample"](s, CROP_W, CROP_H, rng)
        for key in ("left", "right"):
            g = rng.range(0.88, 1.12)
            b = rng.range(-0.05, 0.05)
            c[key] = [min(1.0, max(0.0, v * g + b)) for v in c[key]]
        return c
    if u < 0.80:
        return SN["crop_sample"](G["rend"]._scene(rng), CROP_W, CROP_H, rng)
    if u < 0.92:
        return SN["make_stereogram"](rng, CROP_W, CROP_H, max_disp=36,
                                     n_objects=rng.randint(2, 4))
    return SN["make_textured_scene"](rng, CROP_W, CROP_H, max_disp=36,
                                     n_objects=rng.randint(2, 4))


def grad_task(args):
    seed, flat = args
    SN = G["SN"]
    net = G["net"]
    SN["set_flat_params"](net, flat)
    loss, _, _ = SN["train_grads"](net, draw(SN["Rng"](seed)))
    return loss, SN["flat_grads"](net)


def val_task(args):
    idx, flat = args
    SN, EV = G["SN"], G["EV"]
    if G["val"] is None:
        G["val"] = [s for s in EV["scenes"](os.path.join(HERE, "real"))
                    if s["split"] == "val"]
    sc = G["val"][idx]
    SN["set_flat_params"](G["net"], flat)
    lg, _, W, H = SN["read_image"](sc["left"])
    rg, _, _, _ = SN["read_image"](sc["right"])
    gt, _, non, _, _ = EV["masks"](sc)
    pred, _, _, _ = SN["predict_auto"](G["net"], lg, rg, W, H,
                                       target_peak=TARGET_PEAK, verbose=False)
    return EV["score"](pred, gt, non)[0]


def main(argv):
    steps = int(argv[1]) if len(argv) > 1 else 3000
    start = argv[2] if len(argv) > 2 else os.path.join(HERE, "stereonet_real.txt")
    out = argv[3] if len(argv) > 3 else os.path.join(HERE, "stereonet_v4.txt")
    ncpu = mp.cpu_count()

    SN = {"__name__": "sn"}
    exec(SRC, SN)
    net = build_net(SN, start)
    lr = float(argv[4]) if len(argv) > 4 else 1.0e-3
    opt = SN["Adam"](net.all_params(), lr=lr)
    pool = mp.Pool(ncpu, initializer=init_worker, initargs=(start,))

    def validate():
        flat = SN["flat_params"](net)
        eps = pool.map(val_task, [(i, flat) for i in range(5)])
        return sum(eps) / len(eps), eps

    t0 = time.time()
    v, per = validate()
    print("D=%d, batch %d on %d cores, crops %dx%d. start val EPE %.3f [%s]"
          % (D_NEW, ncpu, ncpu, CROP_W, CROP_H, v,
             " ".join("%.2f" % e for e in per)), flush=True)
    best = v
    SN["save_weights"](net, out)
    run = 0.0
    seed = 1000
    for it in range(1, steps + 1):
        if it == int(steps * 0.6):
            opt.lr *= 0.35
        if it == int(steps * 0.85):
            opt.lr *= 0.35
        flat = SN["flat_params"](net)
        res = pool.map(grad_task, [(seed + k, flat) for k in range(ncpu)])
        seed += ncpu
        n = len(flat)
        avg = [0.0] * n
        for loss, g in res:
            run += loss / ncpu
            for i in range(n):
                avg[i] += g[i]
        inv = 1.0 / ncpu
        SN["set_flat_grads"](net, [a * inv for a in avg])
        opt.step()
        if it % 50 == 0:
            print("step %4d  loss %.4f  (%.1f min)"
                  % (it, run / 50, (time.time() - t0) / 60), flush=True)
            run = 0.0
        if it % 250 == 0 or it == steps:
            v, per = validate()
            tag = ""
            if v < best:
                best = v
                SN["save_weights"](net, out)
                tag = "  <- best, saved"
            print("   VAL EPE %.3f  [%s]%s"
                  % (v, " ".join("%.2f" % e for e in per), tag), flush=True)
            SN["save_weights"](net, out.replace(".txt", "_last.txt"))
    pool.close()
    print("best val %.3f -> %s" % (best, out), flush=True)


if __name__ == "__main__":
    main(sys.argv)
