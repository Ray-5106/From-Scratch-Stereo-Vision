"""Run the Python model on fixed inputs and save inputs + outputs."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'export_weights.py')).read().replace("if __name__ == '__main__':", "if False:"))
net = export(os.path.join(HERE, '..', 'stereonet_weights.txt'), '/tmp/w_d12.json')
cases = []
d, names = SN['read_scene_index'](os.environ.get('RENDERED_DIR', os.path.join(HERE, '..', 'datasets', 'wide')))
s = SN['load_rendered'](d, names[3])
c = SN['crop_sample'](s, 96, 64, SN['Rng'](7))
cases.append(('rendered 96x64, predict', c['left'], c['right'], 96, 64,
              list(SN['predict'](net, c['left'], c['right'], 96, 64))))
lg, _, W, H = SN['read_image'](os.path.join(HERE, '..', 'real/2006/Rocks1/view1.png'))
rg, _, _, _ = SN['read_image'](os.path.join(HERE, '..', 'real/2006/Rocks1/view5.png'))
ls = SN['resize_gray'](lg, W, H, 110, 96); rs = SN['resize_gray'](rg, W, H, 110, 96)
cases.append(('real Rocks1 at 110x96, predict', ls, rs, 110, 96,
              list(SN['predict'](net, ls, rs, 110, 96))))
full = SN['predict_auto'](net, lg, rg, W, H, target_peak=34.0, verbose=False)[0]
cases.append(('real Rocks1 %dx%d, full predict_auto (probe + self-correction)' % (W, H),
              lg, rg, W, H, list(full)))
json.dump([{'name': n, 'left': l, 'right': r, 'W': w, 'H': h, 'ref': ref}
           for (n, l, r, w, h, ref) in cases], open('/tmp/ref_cases.json', 'w'))
print('reference cases written:', [c[0] for c in cases])
