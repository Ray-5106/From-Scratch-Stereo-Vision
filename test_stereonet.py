# =====================================================================
#  Test suite for stereonet.py
#
#      python3 test_stereonet.py
#
#  ZERO IMPORTS, like the library it tests. That constraint has a real
#  consequence here: to test the PNG decoder the suite has to *write* a
#  PNG, which means emitting a zlib stream with a correct Adler-32 and a
#  chunk CRC by hand. Uncompressed DEFLATE blocks make that tractable --
#  a stored block is a length, its complement, and the raw bytes.
#
#  Every hand-derived gradient in the project is checked here against
#  central differences. That is the only reason to trust them: a
#  backward pass that is wrong in a subtle way still trains, just badly,
#  and no loss curve will tell you which.
# =====================================================================

_src = open('stereonet.py').read().replace(
    "if __name__ == '__main__':\n    run_demo()\n", "")
SN = {'__name__': 'stereonet_under_test'}
exec(_src, SN)
globals().update(SN)

PASS = [0]
FAIL = []


def check(name, ok, detail=''):
    PASS[0] += 1
    if ok:
        print('  pass  %-46s %s' % (name, detail))
    else:
        FAIL.append(name)
        print('  FAIL  %-46s %s' % (name, detail))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def relerr(a, b):
    return abs(a - b) / max(1e-9, abs(a) + abs(b))


def numgrad(loss_fn, arr, idx, eps=1e-5):
    old = arr[idx]
    arr[idx] = old + eps
    lp = loss_fn()
    arr[idx] = old - eps
    lm = loss_fn()
    arr[idx] = old
    return (lp - lm) / (2 * eps)


# ---------------------------------------------------------------- 1
def test_arithmetic():
    print('\narithmetic primitives (no math module)')
    check('fsqrt(2)', close(fsqrt(2.0), 1.4142135623730951, 1e-12),
          '%.15f' % fsqrt(2.0))
    check('fsqrt(1e-6)', close(fsqrt(1e-6), 1e-3, 1e-9))
    check('fsqrt(1e8)', close(fsqrt(1e8), 1e4, 1e-9))
    check('fexp(1)', close(fexp(1.0), 2.718281828459045, 1e-12),
          '%.15f' % fexp(1.0))
    check('fexp(-20)', close(fexp(-20.0), 2.0611536224385578e-09, 1e-9))
    check('flog(10)', close(flog(10.0), 2.302585092994046, 1e-12))
    check('flog(fexp(3.7)) round trip', close(flog(fexp(3.7)), 3.7, 1e-10))
    check('ftanh(0.5)', close(ftanh(0.5), 0.46211715726000974, 1e-12))
    check('ftanh saturates', ftanh(50.0) == 1.0 and ftanh(-50.0) == -1.0)
    check('fcos(0)=1', close(fcos(0.0), 1.0, 1e-12))
    check('fcos(pi)= -1', close(fcos(3.141592653589793), -1.0, 1e-10))
    check('fpow(2,10)', close(fpow(2.0, 10.0), 1024.0, 1e-9))


# ---------------------------------------------------------------- 2
def test_rng():
    print('\npseudo-random generator')
    r = Rng(12345)
    vals = [r.uniform() for _ in range(4000)]
    check('uniform in [0,1)', all(0.0 <= v < 1.0 for v in vals))
    mean = sum(vals) / len(vals)
    check('uniform mean near 0.5', abs(mean - 0.5) < 0.02, '%.4f' % mean)
    r2 = Rng(12345)
    check('same seed reproduces', all(close(a, r2.uniform(), 1e-15)
                                      for a in vals[:50]))
    g = [Rng(7).normal() for _ in range(1)]
    r3 = Rng(99)
    ns = [r3.normal() for _ in range(4000)]
    m = sum(ns) / len(ns)
    v = sum((x - m) ** 2 for x in ns) / len(ns)
    check('normal mean near 0', abs(m) < 0.06, '%.4f' % m)
    check('normal variance near 1', abs(v - 1.0) < 0.10, '%.4f' % v)
    ints = [Rng(3).randint(2, 5) for _ in range(1)]
    r4 = Rng(4)
    check('randint within bounds',
          all(2 <= r4.randint(2, 5) <= 5 for _ in range(500)))


# ---------------------------------------------------------------- 3
def test_layer_gradients():
    print('\nlayer gradients vs central differences')
    rng = Rng(99)
    C, H, W = 2, 7, 6
    x = [rng.normal() for _ in range(C * H * W)]

    for label, layer in (('Conv2d stride 2', Conv2d(C, 3, 3, rng, stride=2)),
                         ('Conv2d dilation 2', Conv2d(C, 2, 3, rng, dil=2)),
                         ('Conv2d plain', Conv2d(C, 2, 3, rng))):
        y, OH, OW = layer.forward(x, H, W)
        wts = [1.0 + 0.03 * i for i in range(len(y))]

        def loss():
            yy, _, _ = layer.forward(x, H, W)
            return sum(wts[i] * yy[i] for i in range(len(yy)))

        worst = 0.0
        for pl, gl in layer.params():
            for i in range(0, len(pl), max(1, len(pl) // 4)):
                for _, g in layer.params():
                    for j in range(len(g)):
                        g[j] = 0.0
                loss()
                layer.backward(wts)
                worst = max(worst, relerr(numgrad(loss, pl, i), gl[i]))
        check(label, worst < 1e-6, 'max rel err %.2e' % worst)

    c3 = Conv3d(2, 2, 3, rng)
    D, H3, W3 = 4, 4, 5
    x3 = [rng.normal() for _ in range(2 * D * H3 * W3)]
    y = c3.forward(x3, D, H3, W3)
    wts = [1.0 + 0.01 * i for i in range(len(y))]

    def l3():
        return sum(wts[i] * v for i, v in
                   enumerate(c3.forward(x3, D, H3, W3)))

    worst = 0.0
    for pl, gl in c3.params():
        for i in range(0, len(pl), max(1, len(pl) // 4)):
            for _, g in c3.params():
                for j in range(len(g)):
                    g[j] = 0.0
            l3()
            c3.backward(wts)
            worst = max(worst, relerr(numgrad(l3, pl, i), gl[i]))
    check('Conv3d', worst < 1e-6, 'max rel err %.2e' % worst)


# ---------------------------------------------------------------- 4
def test_stereo_gradients():
    print('\nstereo primitive gradients')
    rng = Rng(21)
    C, H, W, D = 2, 5, 6, 3
    fl = [rng.normal() for _ in range(C * H * W)]
    fr = [rng.normal() for _ in range(C * H * W)]
    cv = CostVolume(D)
    wts = [0.5 + 0.07 * i for i in range(C * D * H * W)]

    def lcv():
        v = cv.forward(fl, fr, C, H, W)
        return sum(wts[i] * v[i] for i in range(len(v)))

    lcv()
    gfl, gfr = cv.backward(wts)
    worst = 0.0
    for i in range(0, C * H * W, 4):
        worst = max(worst, relerr(numgrad(lcv, fl, i), gfl[i]),
                    relerr(numgrad(lcv, fr, i), gfr[i]))
    check('CostVolume both branches', worst < 1e-6,
          'max rel err %.2e' % worst)

    D2, H2, W2 = 5, 4, 4
    cost = [rng.normal() for _ in range(D2 * H2 * W2)]
    sa = SoftArgmin(D2)
    gw = [1.0 + 0.11 * i for i in range(H2 * W2)]

    def lsa():
        d = sa.forward(cost, H2, W2)
        return sum(gw[i] * d[i] for i in range(len(d)))

    lsa()
    gc = sa.backward(gw)
    worst = 0.0
    for i in range(0, len(cost), 3):
        worst = max(worst, relerr(numgrad(lsa, cost, i), gc[i]))
    check('SoftArgmin', worst < 1e-6, 'max rel err %.2e' % worst)

    up = BilinearUpsample()
    xu = [rng.normal() for _ in range(9)]
    gwu = [1.0 + 0.02 * i for i in range(36)]

    def lup():
        y = up.forward(xu, 1, 3, 3, 6, 6)
        return sum(gwu[i] * y[i] for i in range(len(y)))

    lup()
    gxu = up.backward(gwu)
    worst = max(relerr(numgrad(lup, xu, i), gxu[i]) for i in range(9))
    check('BilinearUpsample', worst < 1e-6, 'max rel err %.2e' % worst)

    D3, H3, W3 = 6, 3, 3
    cost = [rng.normal() * 2 for _ in range(D3 * H3 * W3)]
    tgt = [rng.range(0.5, 4.5) for _ in range(H3 * W3)]
    val = [True] * (H3 * W3)
    sa3 = SoftArgmin(D3)

    def lce():
        sa3.forward(cost, H3, W3)
        return disparity_ce(sa3.cache[0], tgt, val, D3, H3, W3)[0]

    sa3.forward(cost, H3, W3)
    _, gce, _ = disparity_ce(sa3.cache[0], tgt, val, D3, H3, W3)
    worst = max(relerr(numgrad(lce, cost, i), gce[i])
                for i in range(0, len(cost), 4))
    check('disparity cross-entropy', worst < 1e-5,
          'max rel err %.2e' % worst)


# ---------------------------------------------------------------- 5
def test_renderer_gradients():
    print('\ndifferentiable splat renderer gradients')
    rng = Rng(777)
    W, H = 20, 16
    cam = Camera.synthetic(W, H, baseline=0.1)
    gc = GaussianCloud()
    for _ in range(5):
        sc = rng.range(0.004, 0.02)
        gc.mx.append(rng.range(-0.3, 0.3))
        gc.my.append(rng.range(-0.25, 0.25))
        gc.mz.append(rng.range(0.6, 1.8))
        gc.sxx.append(sc * rng.range(0.6, 1.4))
        gc.syy.append(sc * rng.range(0.6, 1.4))
        gc.szz.append(sc * 0.2)
        gc.sxy.append(sc * rng.range(-0.2, 0.2))
        gc.sxz.append(sc * rng.range(-0.1, 0.1))
        gc.syz.append(sc * rng.range(-0.1, 0.1))
        gc.cr.append(rng.range(0.1, 0.9))
        gc.cg.append(rng.range(0.1, 0.9))
        gc.cb.append(rng.range(0.1, 0.9))
        gc.raw_op.append(rng.range(-0.5, 1.5))
        gc.raw_sc.append(rng.range(-0.3, 0.3))
    gc.zero_grads()
    r = SplatRenderer(cam, W, H)
    wts = [0.3 + 0.013 * i for i in range(3 * W * H)]

    def loss():
        C = r.render(gc, IDENTITY_R, IDENTITY_T)
        return sum(wts[i] * C[i] for i in range(len(C)))

    gc.zero_grads()
    loss()
    r.backward(wts)
    ana = {'cr': list(gc.g_cr), 'op': list(gc.g_op), 'sc': list(gc.g_sc)}
    worst = 0.0
    for i in range(gc.count()):
        for key, arr in (('cr', gc.cr), ('op', gc.raw_op), ('sc', gc.raw_sc)):
            worst = max(worst, relerr(numgrad(loss, arr, i, 1e-6),
                                      ana[key][i]))
    check('colour, opacity and scale', worst < 1e-5,
          'max rel err %.2e' % worst)


# ---------------------------------------------------------------- 6
def _crc32(data):
    """CRC-32 with a table built on the fly. Needed to write a PNG."""
    table = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (0xEDB88320 ^ (c >> 1)) if (c & 1) else (c >> 1)
        table.append(c)
    c = 0xFFFFFFFF
    for b in data:
        c = table[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _adler32(data):
    a, b = 1, 0
    for byte in data:
        a = (a + byte) % 65521
        b = (b + a) % 65521
    return (b << 16) | a


def _be32(v):
    return bytes([(v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255])


def _chunk(tag, body):
    return _be32(len(body)) + tag + body + _be32(_crc32(tag + body))


def write_png_stored(path, rgb_bytes, W, H):
    """Write an 8-bit RGB PNG using uncompressed DEFLATE blocks.

    A stored block needs no Huffman coding: a header bit, the length,
    its one's complement, and the bytes. That is enough to produce a
    completely standard PNG for the decoder to read back.
    """
    raw = bytearray()
    for y in range(H):
        raw.append(0)                       # filter type None
        raw += rgb_bytes[y * W * 3:(y + 1) * W * 3]
    raw = bytes(raw)

    z = bytearray([0x78, 0x01])             # zlib header, no dictionary
    pos = 0
    while pos < len(raw):
        n = min(65535, len(raw) - pos)
        final = 1 if pos + n >= len(raw) else 0
        z.append(final)                     # BFINAL, BTYPE=00
        z += bytes([n & 255, (n >> 8) & 255,
                    (~n) & 255, ((~n) >> 8) & 255])
        z += raw[pos:pos + n]
        pos += n
    z += _be32(_adler32(raw))

    ihdr = _be32(W) + _be32(H) + bytes([8, 2, 0, 0, 0])
    png = (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr) +
           _chunk(b'IDAT', bytes(z)) + _chunk(b'IEND', b''))
    f = open(path, 'wb')
    f.write(png)
    f.close()


def test_image_io():
    print('\nimage decoding')
    rng = Rng(5150)
    W, H = 37, 23
    rgb = bytes([rng.randint(0, 255) for _ in range(W * H * 3)])
    write_png_stored('/tmp/_t_stored.png', rgb, W, H)
    _, got, gw, gh = read_png('/tmp/_t_stored.png')
    ok = gw == W and gh == H
    worst = 0.0
    for i in range(W * H * 3):
        worst = max(worst, abs(got[i] - rgb[i] / 255.0))
    check('PNG decode (stored blocks)', ok and worst < 1e-6,
          '%dx%d max err %.2e' % (gw, gh, worst))

    gray = [(x * 7 + y * 3) % 256 / 255.0 for y in range(20)
            for x in range(25)]
    write_pgm('/tmp/_t.pgm', gray, 25, 20, 0.0, 1.0)
    g2, _, w2, h2 = read_pnm('/tmp/_t.pgm')
    err = max(abs(a - b) for a, b in zip(gray, g2))
    check('PGM round trip', w2 == 25 and h2 == 20 and err < 1.0 / 255,
          'max err %.4f (8-bit quantisation)' % err)

    rgbf = [0.0] * (25 * 20 * 3)
    for i in range(25 * 20):
        rgbf[3 * i] = gray[i]
        rgbf[3 * i + 1] = 1.0 - gray[i]
        rgbf[3 * i + 2] = 0.5
    write_ppm('/tmp/_t.ppm', rgbf, 25, 20)
    _, r2, _, _ = read_pnm('/tmp/_t.ppm')
    err = max(abs(a - b) for a, b in zip(rgbf, r2))
    check('PPM round trip', err < 1.0 / 255, 'max err %.4f' % err)

    check('read_image dispatches on magic bytes',
          read_image('/tmp/_t_stored.png')[2] == W and
          read_image('/tmp/_t.pgm')[2] == 25)


# ---------------------------------------------------------------- 7
def test_geometry():
    print('\ngeometry')
    rng = Rng(4242)
    pts = [(rng.range(0, 100), rng.range(0, 100)) for _ in range(160)]
    tris = delaunay(pts)
    viol = 0
    for (i, j, k) in tris:
        cc = _circumcircle(pts[i][0], pts[i][1], pts[j][0], pts[j][1],
                           pts[k][0], pts[k][1])
        if cc is None:
            continue
        ux, uy, r2 = cc
        for m in range(len(pts)):
            if m in (i, j, k):
                continue
            dx = pts[m][0] - ux
            dy = pts[m][1] - uy
            if dx * dx + dy * dy < r2 - 1e-7:
                viol += 1
    check('Delaunay empty-circumcircle property', viol == 0,
          '%d triangles, %d violations' % (len(tris), viol))

    sa = SoftArgmin(5)
    cost = [abs(d - 3) * 4.0 for d in range(5)]
    d = sa.forward(cost, 1, 1)[0]
    check('soft-argmin finds a V minimum', abs(d - 3.0) < 0.01,
          'regressed %.4f for a minimum at 3' % d)

    up = BilinearUpsample()
    flat = [2.5] * 16
    r = up.forward(flat, 1, 4, 4, 9, 9)
    check('bilinear preserves a constant field',
          max(r) - min(r) < 1e-12, 'range %.1e' % (max(r) - min(r)))

    cam = Camera(100.0, 32.0, 24.0, 0.1)
    z = cam.depth_from_disparity(10.0)
    X, Y, Z = cam.unproject(32.0, 24.0, z)
    check('unprojection at the principal point is on axis',
          abs(X) < 1e-9 and abs(Y) < 1e-9 and close(Z, 1.0, 1e-9),
          'Z=%.4f m for d=10' % Z)

    cv = CostVolume(6)
    rng2 = Rng(11)
    base = [rng2.uniform() for _ in range(40)]
    shift = 3
    fL = [base[x + 8] for x in range(24)]
    fR = [base[x + 8 + shift] for x in range(24)]
    v = cv.forward(fL, fR, 1, 1, 24)
    sums = [sum(v[d * 24 + x] for x in range(8, 22)) for d in range(6)]
    best = sums.index(min(sums))
    check('cost volume minimises at the true shift', best == shift,
          'argmin %d, truth %d, cost %.4f' % (best, shift, sums[shift]))


# ---------------------------------------------------------------- 8
def test_consistency_and_checkpoint():
    print('\nconsistency check and checkpoints')
    rng = Rng(5)
    W, H = 9, 4
    a = [rng.uniform() for _ in range(W * H)]
    b = mirror_h(mirror_h(a, 1, W, H), 1, W, H)
    check('horizontal mirror is an involution',
          all(abs(x - y) < 1e-12 for x, y in zip(a, b)))

    net = StereoNet(Rng(1), feat_C=6, cost_C=3, D=6, max_disp=24,
                    refine_C=5, refine_dilations=(1, 2))
    save_weights(net, '/tmp/_t_ck.txt')
    cfg = read_checkpoint_config('/tmp/_t_ck.txt')
    check('checkpoint records its architecture',
          cfg is not None and cfg['feat_C'] == 6 and cfg['refine_C'] == 5,
          str(cfg))
    net2 = load_model('/tmp/_t_ck.txt')
    worst = 0.0
    for (pa, _), (pb, _) in zip(net.all_params(), net2.all_params()):
        for x, y in zip(pa, pb):
            worst = max(worst, abs(x - y))
    check('checkpoint round trip', worst < 1e-7,
          'max abs diff %.2e (9 significant digits in text)' % worst)


# ---------------------------------------------------------------- 9
def test_learning():
    print('\nend to end: the network actually learns')
    rng = Rng(2024)
    net = StereoNet(rng, feat_C=6, cost_C=3, D=6, max_disp=24, refine_C=4,
                    refine_dilations=(1, 2))
    opt = Adam(net.all_params(), lr=3.0e-3)
    scene = make_stereogram(Rng(9), 32, 32, max_disp=8, n_objects=2)
    first = None
    for it in range(25):
        loss, pred, tgt = train_step(net, opt, scene)
        if first is None:
            first = loss
    check('loss falls on a single scene', loss < first * 0.6,
          '%.3f -> %.3f over 25 steps' % (first, loss))

    epe, b1, b3 = metrics(pred, tgt['d'][2], tgt['v'][2])
    check('EPE is finite and sane', epe == epe and epe < 40.0,
          'EPE %.2f px' % epe)


def test_zero_imports():
    """The central claim of the project, checked the way a sceptic
    would check it: no line of either file may begin with 'import' or
    'from', docstrings and comments included. A docstring sentence that
    happened to start with 'from' once tripped exactly this, and a grep
    cannot tell prose from code."""
    print('\nthe zero-import claim')
    for name in ('stereonet.py', 'test_stereonet.py'):
        f = open(name, 'r')
        bad = []
        n = 0
        for line in f:
            n += 1
            t = line.lstrip()
            if t.startswith('import ') or t.startswith('from '):
                bad.append(n)
        f.close()
        check('%s has no import/from lines' % name, not bad,
              '%d lines scanned%s' % (n, (', offending: %s' % bad) if bad
                                      else ''))


def test_refinement():
    print('\n[full-resolution refinement, planes, background layer]')
    W, H = 96, 64
    s = make_textured_scene(Rng(5), W, H, max_disp=12, n_objects=3)
    L, Rt, gt = s['left'], s['right'], s['disp']
    valid = s.get('valid')
    zero = census(L, W, H, 4, 3)
    check('census: identical images differ in 0 bits',
          sum(_popcount(a ^ a) for a in zero) == 0)
    # a coarse estimate 1-2 px off everywhere, the refinement's job
    rng = Rng(9)
    d0 = [g + rng.range(-2.0, 2.0) for g in gt]

    def epe(d):
        n = 0
        e = 0.0
        for i in range(W * H):
            x = i % W
            if x < 14 or (valid is not None and not valid[i]):
                continue
            e += abs(d[i] - gt[i])
            n += 1
        return e / n
    e0 = epe(d0)
    d1 = refine_fullres(L, Rt, d0, W, H, R=4, P1=12.0, P2=150.0)
    e1 = epe(d1)
    check('band SGM cuts a 1-2 px error by more than half', e1 < 0.5 * e0,
          '%.3f -> %.3f px' % (e0, e1))
    d2, st = lr_refine(L, Rt, d0, W, H, tol=2.0, R=4, P1=12.0, P2=150.0)
    check('LR check marks the unmatched left margin occluded',
          all(st[y * W] == 1 for y in range(H) if gt[y * W] >= 1.5))
    # plane snapping: a slanted plane plus small noise and a few outliers
    rng = Rng(3)
    plane = [0.05 * (i % W) + 0.02 * (i // W) + 6.0 for i in range(W * H)]
    noisy = [p + rng.range(-0.3, 0.3) for p in plane]
    for k in range(0, W * H, 97):
        noisy[k] += 5.0
    grey = [0.5] * (3 * W * H)
    snapped = plane_snap(noisy, grey, W, H)
    inl = [i for i in range(W * H) if i % 97]
    en = sum(abs(noisy[i] - plane[i]) for i in inl) / len(inl)
    es = sum(abs(snapped[i] - plane[i]) for i in inl) / len(inl)
    check('plane snap removes the noise on a slanted plane', es < 0.25 * en,
          'mean error %.3f -> %.3f px' % (en, es))
    check('plane snap leaves pixels off the plane alone',
          all(snapped[k] == noisy[k] for k in range(0, W * H, 97)))
    # background region: a near square in front of a far wall
    step = [4.0] * (W * H)
    for y in range(20, 44):
        for x in range(40, 60):
            step[y * W + x] = 16.0
    mask, m, src = background_region(step, W, H)
    inside = [y * W + x for y in range(20, 44) for x in (40, 41, 58, 59)]
    check('background layer covers the square\'s edges',
          all(mask[i] for i in inside))
    check('background layer continues the wall behind the square',
          all(abs(m[i] - 4.0) < 1e-9 for i in inside),
          'layer disparity %.2f' % m[inside[0]])
    check('background layer leaves the open wall alone',
          not mask[5 * W + 5] and not mask[60 * W + 90])
    cam = Camera.synthetic(W, H)
    g = grid_mesh(step, W, H, cam, step=2, fill=False, min_disp=0.5)
    crosses = 0
    for t in range(0, len(g['idx']), 3):
        ds = [g['disp'][g['idx'][t + k]] for k in range(3)]
        if max(ds) - min(ds) > 5.0:
            crosses += 1
    check('viewer grid never bridges a depth edge', crosses == 0,
          '%d vertices, %d triangles' % (len(g['disp']), g['main_tris']))
    k = len(g['disp']) // 2
    x = int(g['uv'][2 * k] * W)
    y = int(g['uv'][2 * k + 1] * H)
    P = cam.unproject(x, y, cam.depth_from_disparity(g['disp'][k]))
    check('viewer grid texture coordinates sit on their own pixel',
          abs(P[0] - g['pos'][3 * k]) < 1e-9 and abs(P[1] - g['pos'][3 * k + 1]) < 1e-9)


def main():
    print('=' * 72)
    print('stereonet.py test suite')
    print('=' * 72)
    test_zero_imports()
    test_arithmetic()
    test_rng()
    test_layer_gradients()
    test_stereo_gradients()
    test_renderer_gradients()
    test_image_io()
    test_geometry()
    test_consistency_and_checkpoint()
    test_learning()
    test_refinement()
    print('\n' + '=' * 72)
    if FAIL:
        print('%d of %d checks FAILED:' % (len(FAIL), PASS[0]))
        for f in FAIL:
            print('   - ' + f)
    else:
        print('all %d checks passed' % PASS[0])
    print('=' * 72)


main()
