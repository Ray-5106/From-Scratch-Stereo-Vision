"""
Widen a trained checkpoint without changing what it computes (Net2Net-style).

    python3 widen_net.py <in_ckpt> <out_ckpt> <feat_C> <cost_C> <refine_C>

HARNESS, NOT THE MODEL. Every convolution of the wider network receives
the old weights in its top-left block. Weights that read the new
channels are set to zero, so at the start the wider network produces
exactly the old output, while the new channels (randomly initialised on
their input side) receive gradient from the first step.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SN = {"__name__": "sn"}
exec(open(os.path.join(HERE, "stereonet.py")).read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", ""), SN)

src, dst, fC, cC, rC = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
old = SN["load_model"](src)
cfg = SN["read_checkpoint_config"](src)
new = SN["StereoNet"](SN["Rng"](77), feat_C=fC, cost_C=cC, D=old.D, max_disp=old.max_disp,
                      refine_C=rC, refine_dilations=cfg.get("refine_dil", (1, 2, 4)))
new.target_peak = old.target_peak
convs = lambda n: [m for m in n.modules() if hasattr(m, "ic")]
co, cn = convs(old), convs(new)
assert len(co) == len(cn)
for a, b in zip(co, cn):
    kk = len(a.w) // (a.oc * a.ic)          # k*k or k*k*k
    assert kk == len(b.w) // (b.oc * b.ic)
    for o in range(b.oc):
        for i in range(b.ic):
            for t in range(kk):
                j = (o * b.ic + i) * kk + t
                if o < a.oc and i < a.ic:
                    b.w[j] = a.w[(o * a.ic + i) * kk + t]
                elif o < a.oc:
                    b.w[j] = 0.0            # old outputs ignore new channels
                # new outputs keep their random input weights
    for o in range(b.oc):
        b.b[o] = a.b[o] if o < a.oc else 0.0
SN["save_weights"](new, dst)
print("widened %s -> %s: %d -> %d parameters" % (src, dst,
      sum(len(p) for p, _ in old.all_params()), sum(len(p) for p, _ in new.all_params())))
