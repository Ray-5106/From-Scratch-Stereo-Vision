"""
Assemble the single-file web demo.

    python3 build_web.py <checkpoint> <D> <target_peak> <test_epe> <out.html>

Harness, not model: numpy/PIL read the example images, node runs the
verified JavaScript port to precompute each example's result.
"""
import base64, json, os, subprocess, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'real')
exec(open(os.path.join(HERE, 'export_weights.py')).read().replace("if __name__ == '__main__':", "if False:"))

ckpt, D, T, test_epe, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
wjson = '/tmp/web_weights.json'
net = export(ckpt, wjson, D=D)
params = sum(len(p) for p, _ in net.all_params())

SAMPLES = [
    ('cones', 'Cones', '2003/cones', 'im2.png', 'im6.png', 'disp2.png', 'disp6.png', 4.0),
    ('teddy', 'Teddy', '2003/teddy', 'im2.png', 'im6.png', 'disp2.png', 'disp6.png', 4.0),
    ('art', 'Art', '2005/Art', 'view1.png', 'view5.png', 'disp1.png', 'disp5.png', 3.0),
]
uri = lambda p: 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode()
def gray(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float64) / 255.0
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
def rgbf(p):
    return (np.asarray(Image.open(p).convert('RGB'), dtype=np.float64) / 255.0).ravel()

cases = []
for key, label, d, l, r, gl, gr, sc in SAMPLES:
    g1, g2 = gray(os.path.join(ROOT, d, l)), gray(os.path.join(ROOT, d, r))
    cases.append({'key': key, 'W': g1.shape[1], 'H': g1.shape[0],
                  'left': g1.ravel().tolist(), 'right': g2.ravel().tolist(),
                  'rgb': rgbf(os.path.join(ROOT, d, l)).tolist()})
json.dump(cases, open('/tmp/web_cases.json', 'w'))
subprocess.run(['node', os.path.join(HERE, 'precompute.js'), wjson, str(T),
                '/tmp/web_cases.json', '/tmp/web_pred.json'], check=True)
pred = json.load(open('/tmp/web_pred.json'))

samples = {}
for key, label, d, l, r, gl, gr, sc in SAMPLES:
    enc = lambda a: base64.b64encode(np.clip(np.round(np.array(a) * 64), 0, 65535).astype('>u2').tobytes()).decode()
    samples[key] = {'label': label, 'scale': sc,
                    'left': uri(os.path.join(ROOT, d, l)), 'right': uri(os.path.join(ROOT, d, r)),
                    'gtl': uri(os.path.join(ROOT, d, gl)), 'gtr': uri(os.path.join(ROOT, d, gr)),
                    'raw': enc(pred[key]['raw']), 'ref': enc(pred[key]['ref'])}

chips = ''.join('<button class="chip" data-k="%s" aria-pressed="false">%s</button>' % (k, s['label'])
                for k, s in samples.items())
html = open(os.path.join(HERE, 'page_template.html')).read()
rep = {'__INFER_JS__': open(os.path.join(HERE, 'stereonet_infer.js')).read(),
       '__RECON_JS__': open(os.path.join(HERE, 'recon.js')).read(),
       '__REFINE_JS__': open(os.path.join(HERE, 'refine.js')).read(),
       '__WEIGHTS__': open(wjson).read(), '__SAMPLES__': json.dumps(samples),
       '__TARGET_PEAK__': repr(T), '__PARAMS__': '{:,}'.format(params),
       '__TESTEPE__': test_epe, '__VERIFY__': os.environ.get('VERIFY_TXT', '4&times;10<sup>&minus;14</sup>'),
       '__DMAX__': str(D * 4), '__CHIPS__': chips}
for k, v in rep.items():
    html = html.replace(k, v)
open(out, 'w').write(html)
print('built %s: %.2f MB, %d params, D=%d, target peak %.0f' % (out, os.path.getsize(out) / 1e6, params, D, T))
