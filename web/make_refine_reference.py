"""Run the Python refinement on a real pair and save inputs + outputs for
verify_refine.js. Harness, not model."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SN = {"__name__": "sn"}
exec(open(os.path.join(HERE, "..", "stereonet.py")).read().replace("if __name__ == '__main__':\n    run_demo()\n", ""), SN)
scene = sys.argv[1] if len(sys.argv) > 1 else "2003/cones"
L, R = ("im2.png", "im6.png") if scene.startswith("2003") else ("view1.png", "view5.png")
lg, lrgb, W, H = SN["read_image"](os.path.join(HERE, "..", "real", scene, L))
rg, _, _, _ = SN["read_image"](os.path.join(HERE, "..", "real", scene, R))
net = SN["load_model"](os.path.join(HERE, "..", "stereonet_weights.txt"))
d0 = SN["predict_auto"](net, lg, rg, W, H, verbose=False)[0]
d, st = SN["refine_disparity"](lg, rg, lrgb, d0, W, H)
reg = SN["background_region"](d, W, H)
json.dump({"scene": scene, "W": W, "H": H, "left": lg, "right": rg, "rgb": lrgb, "disp0": d0,
           "ref": d, "status": st, "mask": [1 if b else 0 for b in reg[0]], "bgm": reg[1], "src": reg[2]}, open("/tmp/refine_ref.json", "w"))
print("reference written for", scene, W, H)
