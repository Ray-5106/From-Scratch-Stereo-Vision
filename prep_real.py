"""
Prepare real Middlebury training samples once, for the PyPy trainer.

    python3 prep_real.py  [peaks...]      (runs under CPython: needs numpy)

Writes real_train_samples.pkl. PyPy has no numpy, so the image
resampling is done here and the result handed over as plain Python lists.
Only the 'train' split is touched.
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
ft = {"__name__": "ft", "__file__": os.path.join(HERE, "finetune_real.py")}
exec(open(os.path.join(HERE, "finetune_real.py")).read().replace(
    'if __name__ == "__main__":\n    main(sys.argv)', ""), ft)

peaks = tuple(float(a) for a in sys.argv[1:]) or (22.0, 30.0, 38.0)
ft["PEAKS"] = peaks
train = [s for s in ft["EV"]["scenes"](ft["ROOT"]) if s["split"] == "train"]
out = []
for sc in train:
    for smp in ft["prepare"](sc):
        smp["left_rgb"] = None            # not needed for training
        if smp["W"] >= 96 and smp["H"] >= 64:
            out.append(smp)
with open(os.path.join(HERE, "real_train_samples.pkl"), "wb") as f:
    pickle.dump(out, f, protocol=2)
print("%d samples at peaks %s from %d training scenes"
      % (len(out), peaks, len(train)))
