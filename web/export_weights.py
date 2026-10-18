"""Export a checkpoint to JSON for the browser port. Harness, not model."""
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '..', 'stereonet.py')).read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", "")
SN = {'__name__': 'sn'}; exec(src, SN)

def export(ckpt, out, D=None):
    cfg = SN['read_checkpoint_config'](ckpt)
    net = SN['StereoNet'](SN['Rng'](1), feat_C=cfg['feat_C'], cost_C=cfg['cost_C'],
                          D=D or cfg['D'], max_disp=cfg['max_disp'],
                          refine_C=cfg['refine_C'], refine_dilations=cfg['refine_dil'])
    SN['load_weights'](net, ckpt)
    blocks = [[float('%.9g' % v) for v in p] for p, _ in net.all_params()]
    conf = dict(feat_C=cfg['feat_C'], cost_C=cfg['cost_C'], D=D or cfg['D'],
                refine_C=cfg['refine_C'], refine_dil=list(cfg['refine_dil']))
    json.dump({'config': conf, 'blocks': blocks}, open(out, 'w'))
    return net

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
