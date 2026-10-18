# =====================================================================
#  StereoNet : Guided Hierarchical Refinement for Real-Time Edge-Aware
#              Depth Prediction  (Khamis et al., ECCV 2018)
#
#  Implemented from scratch in pure Python.
#  ZERO import statements. No numpy, no torch, no PIL, no math module.
#
#  Everything lives in this one file because splitting the code across
#  modules would itself require an import statement.
#
#  Section index
#    1. Arithmetic primitives        (sqrt, exp, log, tanh, pow)
#    2. Pseudo-random number generator
#    3. Tensor helpers               (flat-list tensors + index math)
#    4. Image I/O                    (PGM / PPM read + write, ASCII view)
#    5. Layers                       (conv2d, conv3d, activations)
#    6. Stereo primitives            (cost volume, soft-argmin, upsample)
#    7. The network                  (feature net, cost filter, refinement)
#    8. Loss and optimiser           (pseudo-Huber, Adam)
#    9. Synthetic data generator     (random-dot stereograms)
#   10. Training / evaluation / demo
# =====================================================================


# =====================================================================
#  SECTION 1 : ARITHMETIC PRIMITIVES
#  The math module is an import, so these are built by hand.
# =====================================================================

LN2   = 0.6931471805599453
SQRT2 = 1.4142135623730951


def fsqrt(x):
    """Square root by Newton-Raphson on f(y) = y*y - x."""
    if x <= 0.0:
        return 0.0
    # Seed via exponent halving so convergence is fast for any magnitude.
    y = x
    if y > 4.0:
        s = 1.0
        while y > 4.0:
            y *= 0.25
            s *= 2.0
        y = s
    elif y < 0.25:
        s = 1.0
        while y < 0.25:
            y *= 4.0
            s *= 0.5
        y = s
    else:
        y = 1.0
    for _ in range(24):
        y = 0.5 * (y + x / y)
    return y


def fexp(x):
    """e**x via range reduction to [-ln2/2, ln2/2] plus a Taylor series."""
    if x > 700.0:
        return 1.0e308
    if x < -700.0:
        return 0.0
    # x = k*ln2 + r
    k = int(x / LN2 + (0.5 if x >= 0.0 else -0.5))
    r = x - k * LN2
    # Taylor series for e**r, r is small so this converges in ~12 terms.
    term = 1.0
    total = 1.0
    for n in range(1, 16):
        term = term * r / n
        total += term
    # Multiply by 2**k without using pow().
    if k >= 0:
        for _ in range(k):
            total *= 2.0
    else:
        for _ in range(-k):
            total *= 0.5
    return total


def flog(x):
    """Natural log via atanh series: ln(x) = 2*atanh((x-1)/(x+1))."""
    if x <= 0.0:
        return -1.0e308
    k = 0
    while x > 1.5:
        x *= 0.5
        k += 1
    while x < 0.5:
        x *= 2.0
        k -= 1
    z = (x - 1.0) / (x + 1.0)
    z2 = z * z
    total = 0.0
    term = z
    n = 1
    while n < 40:
        total += term / n
        term *= z2
        n += 2
    return 2.0 * total + k * LN2


def ftanh(x):
    """tanh, written so large |x| cannot overflow."""
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    e = fexp(2.0 * x)
    return (e - 1.0) / (e + 1.0)


def fpow(base, e):
    """base**e for positive base."""
    if base <= 0.0:
        return 0.0
    return fexp(e * flog(base))


def fabs(x):
    return -x if x < 0.0 else x


def fmax(a, b):
    return a if a > b else b


def fmin(a, b):
    return a if a < b else b


def fclamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# =====================================================================
#  SECTION 2 : PSEUDO-RANDOM NUMBER GENERATOR
#  The random module is an import, so here is an xorshift64* generator.
# =====================================================================

class Rng:
    """xorshift64* generator. Small and fast; adequate for weight init."""

    MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed=88172645463325252):
        if seed == 0:
            seed = 88172645463325252
        self.s = seed & Rng.MASK

    def next_u64(self):
        s = self.s
        s ^= (s << 13) & Rng.MASK
        s ^= (s >> 7)
        s ^= (s << 17) & Rng.MASK
        self.s = s
        return (s * 2685821657736338717) & Rng.MASK

    def uniform(self):
        """Uniform in [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def range(self, lo, hi):
        """Uniform float in [lo, hi)."""
        return lo + (hi - lo) * self.uniform()

    def randint(self, lo, hi):
        """Uniform integer in [lo, hi] inclusive."""
        if hi <= lo:
            return lo
        return lo + int(self.next_u64() % (hi - lo + 1))

    def normal(self):
        """Standard normal via Box-Muller (needs our own sqrt/log/cos)."""
        u1 = self.uniform()
        if u1 < 1.0e-12:
            u1 = 1.0e-12
        u2 = self.uniform()
        radius = fsqrt(-2.0 * flog(u1))
        return radius * fcos(6.283185307179586 * u2)


def fcos(x):
    """Cosine by range reduction plus Taylor series."""
    TWO_PI = 6.283185307179586
    # Reduce into [-pi, pi].
    while x > 3.141592653589793:
        x -= TWO_PI
    while x < -3.141592653589793:
        x += TWO_PI
    x2 = x * x
    term = 1.0
    total = 1.0
    for n in range(1, 12):
        term = term * (-x2) / ((2 * n - 1) * (2 * n))
        total += term
    return total


# =====================================================================
#  SECTION 3 : TENSOR HELPERS
#  Tensors are flat Python lists plus an explicit shape tuple. Flat
#  lists with hand-computed strides are several times faster than
#  nested lists, which matters a great deal without numpy.
# =====================================================================

def zeros(n):
    return [0.0] * n


def tensor_add_inplace(dst, src):
    for i in range(len(dst)):
        dst[i] += src[i]


def tensor_scale(t, k):
    return [v * k for v in t]


def tensor_sum(t):
    total = 0.0
    for v in t:
        total += v
    return total


# =====================================================================
#  SECTION 4 : IMAGE I/O
#  PIL is an import, so PGM/PPM (Netpbm) files are parsed by hand.
#  Netpbm is chosen because its container is plain and uncompressed:
#  a magic number, dimensions, a maxval, then raw samples. A PNG would
#  require writing a DEFLATE decoder, which is not where the day should
#  go.
# =====================================================================

def _pnm_tokens(data, start):
    """Yield header tokens from a Netpbm file, skipping '#' comments."""
    toks = []
    i = start
    n = len(data)
    while len(toks) < 3 and i < n:
        ch = data[i]
        if ch == 35:                      # '#'
            while i < n and data[i] not in (10, 13):
                i += 1
        elif ch in (32, 9, 10, 13):       # whitespace
            i += 1
        else:
            j = i
            while j < n and data[j] not in (32, 9, 10, 13):
                j += 1
            toks.append(data[i:j].decode('ascii'))
            i = j
    return toks, i


def read_pnm(path):
    """Read a PGM or PPM file.

    Returns (gray, rgb, width, height) where gray is a flat list of
    floats in [0,1] and rgb is a flat list of 3*W*H floats in [0,1]
    (a grey image is expanded to three equal channels).
    """
    f = open(path, 'rb')
    data = f.read()
    f.close()

    magic = data[0:2].decode('ascii')
    if magic not in ('P2', 'P3', 'P5', 'P6'):
        raise ValueError('not a PGM/PPM file: ' + path)

    toks, pos = _pnm_tokens(data, 2)
    width = int(toks[0])
    height = int(toks[1])
    maxval = int(toks[2])
    inv = 1.0 / maxval

    channels = 3 if magic in ('P3', 'P6') else 1
    count = width * height * channels
    samples = [0.0] * count

    if magic in ('P2', 'P3'):
        # ASCII payload.
        body = data[pos:].split()
        for i in range(count):
            samples[i] = int(body[i]) * inv
    else:
        # Binary payload: exactly one whitespace byte after maxval.
        pos += 1
        if maxval < 256:
            for i in range(count):
                samples[i] = data[pos + i] * inv
        else:
            for i in range(count):
                hi = data[pos + 2 * i]
                lo = data[pos + 2 * i + 1]
                samples[i] = ((hi << 8) | lo) * inv

    npix = width * height
    if channels == 3:
        rgb = samples
        gray = [0.0] * npix
        for i in range(npix):
            j = 3 * i
            gray[i] = 0.299 * rgb[j] + 0.587 * rgb[j + 1] + 0.114 * rgb[j + 2]
    else:
        gray = samples
        rgb = [0.0] * (3 * npix)
        for i in range(npix):
            v = gray[i]
            j = 3 * i
            rgb[j] = v
            rgb[j + 1] = v
            rgb[j + 2] = v

    return gray, rgb, width, height


def write_pgm(path, gray, width, height, lo=None, hi=None):
    """Write a binary P5 greyscale image, auto-scaling unless told not to."""
    if lo is None:
        lo = gray[0]
        for v in gray:
            if v < lo:
                lo = v
    if hi is None:
        hi = gray[0]
        for v in gray:
            if v > hi:
                hi = v
    span = hi - lo
    if span <= 1.0e-12:
        span = 1.0

    out = bytearray()
    out += ('P5\n%d %d\n255\n' % (width, height)).encode('ascii')
    for v in gray:
        t = int(255.0 * (v - lo) / span + 0.5)
        if t < 0:
            t = 0
        elif t > 255:
            t = 255
        out.append(t)

    f = open(path, 'wb')
    f.write(bytes(out))
    f.close()


def write_ppm(path, rgb, width, height):
    """Write a binary P6 colour image from floats in [0,1]."""
    out = bytearray()
    out += ('P6\n%d %d\n255\n' % (width, height)).encode('ascii')
    for v in rgb:
        t = int(255.0 * v + 0.5)
        if t < 0:
            t = 0
        elif t > 255:
            t = 255
        out.append(t)
    f = open(path, 'wb')
    f.write(bytes(out))
    f.close()


ASCII_RAMP = ' .:-=+*#%@'


def ascii_view(gray, width, height, out_cols=64, invert=False):
    """Render a field as ASCII so results are visible with no image viewer."""
    lo = gray[0]
    hi = gray[0]
    for v in gray:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    span = hi - lo
    if span <= 1.0e-12:
        span = 1.0

    if out_cols > width:
        out_cols = width
    out_rows = int(height * out_cols / float(width) * 0.5)
    if out_rows < 1:
        out_rows = 1

    levels = len(ASCII_RAMP) - 1
    lines = []
    for r in range(out_rows):
        y = int(r * height / out_rows)
        row = []
        for c in range(out_cols):
            x = int(c * width / out_cols)
            t = (gray[y * width + x] - lo) / span
            if invert:
                t = 1.0 - t
            idx = int(t * levels + 0.5)
            if idx < 0:
                idx = 0
            elif idx > levels:
                idx = levels
            row.append(ASCII_RAMP[idx])
        lines.append(''.join(row))
    return '\n'.join(lines)


def read_calib(path):
    """Parse a Middlebury-style calib.txt for focal length and baseline.

    Recognises lines of the form
        cam0=[f 0 cx; 0 f cy; 0 0 1]
        baseline=193.001
        doffs=131.111
    Returns a dict with fx, cx, cy, baseline, doffs. Missing entries are
    left as None so the caller can substitute defaults.
    """
    out = {'fx': None, 'cx': None, 'cy': None,
           'baseline': None, 'doffs': 0.0}
    f = open(path, 'r')
    for line in f:
        line = line.strip()
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        if key == 'cam0':
            body = val.replace('[', '').replace(']', '')
            rows = body.split(';')
            first = rows[0].split()
            out['fx'] = float(first[0])
            out['cx'] = float(first[2])
            if len(rows) > 1:
                second = rows[1].split()
                out['cy'] = float(second[2])
        elif key == 'baseline':
            out['baseline'] = float(val.strip())
        elif key == 'doffs':
            out['doffs'] = float(val.strip())
    f.close()
    return out


# =====================================================================
#  SECTION 5 : LAYERS
#  Every layer owns its parameters, its gradients, and a cache from the
#  forward pass. Backward is written out by hand. This is the part
#  that makes the project a network rather than block matching.
# =====================================================================

class Conv2d:
    """2D convolution with stride, zero padding and dilation.

    Weight layout : [oc][ic][ky][kx] flattened
    Data layout   : [c][y][x] flattened
    """

    def __init__(self, ic, oc, k, rng, stride=1, pad=None, dil=1):
        self.ic = ic
        self.oc = oc
        self.k = k
        self.stride = stride
        self.dil = dil
        self.pad = (dil * (k - 1)) // 2 if pad is None else pad

        fan_in = ic * k * k
        std = fsqrt(2.0 / fan_in)            # He initialisation
        n = oc * ic * k * k
        self.w = [rng.normal() * std for _ in range(n)]
        self.b = [0.0] * oc
        self.gw = [0.0] * n
        self.gb = [0.0] * oc
        self.cache = None

    def out_size(self, H, W):
        eff = self.dil * (self.k - 1) + 1
        OH = (H + 2 * self.pad - eff) // self.stride + 1
        OW = (W + 2 * self.pad - eff) // self.stride + 1
        return OH, OW

    def forward(self, x, H, W):
        ic, oc, k = self.ic, self.oc, self.k
        s, p, d = self.stride, self.pad, self.dil
        OH, OW = self.out_size(H, W)
        w, b = self.w, self.b
        HW = H * W
        kk = k * k
        ickk = ic * kk

        y = [0.0] * (oc * OH * OW)
        for o in range(oc):
            wo = o * ickk
            ob = o * OH * OW
            bias = b[o]
            for oy in range(OH):
                iy0 = oy * s - p
                orow = ob + oy * OW
                for ox in range(OW):
                    ix0 = ox * s - p
                    acc = bias
                    for c in range(ic):
                        xb = c * HW
                        wc = wo + c * kk
                        for ky in range(k):
                            iy = iy0 + ky * d
                            if iy < 0 or iy >= H:
                                continue
                            xr = xb + iy * W
                            wr = wc + ky * k
                            for kx in range(k):
                                ix = ix0 + kx * d
                                if ix < 0 or ix >= W:
                                    continue
                                acc += x[xr + ix] * w[wr + kx]
                    y[orow + ox] = acc

        self.cache = (x, H, W, OH, OW)
        return y, OH, OW

    def backward(self, gy):
        x, H, W, OH, OW = self.cache
        ic, oc, k = self.ic, self.oc, self.k
        s, p, d = self.stride, self.pad, self.dil
        w = self.w
        gw, gb = self.gw, self.gb
        HW = H * W
        kk = k * k
        ickk = ic * kk

        gx = [0.0] * (ic * HW)
        for o in range(oc):
            wo = o * ickk
            ob = o * OH * OW
            gsum = 0.0
            for oy in range(OH):
                iy0 = oy * s - p
                orow = ob + oy * OW
                for ox in range(OW):
                    g = gy[orow + ox]
                    if g == 0.0:
                        continue
                    gsum += g
                    ix0 = ox * s - p
                    for c in range(ic):
                        xb = c * HW
                        wc = wo + c * kk
                        for ky in range(k):
                            iy = iy0 + ky * d
                            if iy < 0 or iy >= H:
                                continue
                            xr = xb + iy * W
                            wr = wc + ky * k
                            for kx in range(k):
                                ix = ix0 + kx * d
                                if ix < 0 or ix >= W:
                                    continue
                                gw[wr + kx] += x[xr + ix] * g
                                gx[xr + ix] += w[wr + kx] * g
            gb[o] += gsum
        return gx

    def params(self):
        return [(self.w, self.gw), (self.b, self.gb)]


class Conv3d:
    """3D convolution over a cost volume, stride 1, 'same' padding.

    Weight layout : [oc][ic][kd][ky][kx] flattened
    Data layout   : [c][d][y][x] flattened
    """

    def __init__(self, ic, oc, k, rng):
        self.ic = ic
        self.oc = oc
        self.k = k
        self.pad = k // 2

        fan_in = ic * k * k * k
        std = fsqrt(2.0 / fan_in)
        n = oc * ic * k * k * k
        self.w = [rng.normal() * std for _ in range(n)]
        self.b = [0.0] * oc
        self.gw = [0.0] * n
        self.gb = [0.0] * oc
        self.cache = None

    def forward(self, x, D, H, W):
        ic, oc, k, p = self.ic, self.oc, self.k, self.pad
        w, b = self.w, self.b
        HW = H * W
        DHW = D * HW
        kkk = k * k * k
        kk = k * k
        ickkk = ic * kkk

        y = [0.0] * (oc * DHW)
        for o in range(oc):
            wo = o * ickkk
            ob = o * DHW
            bias = b[o]
            for od in range(D):
                d0 = od - p
                for oy in range(H):
                    y0 = oy - p
                    orow = ob + od * HW + oy * W
                    for ox in range(W):
                        x0 = ox - p
                        acc = bias
                        for c in range(ic):
                            xb = c * DHW
                            wc = wo + c * kkk
                            for kd in range(k):
                                dd = d0 + kd
                                if dd < 0 or dd >= D:
                                    continue
                                xd = xb + dd * HW
                                wd = wc + kd * kk
                                for ky in range(k):
                                    yy = y0 + ky
                                    if yy < 0 or yy >= H:
                                        continue
                                    xr = xd + yy * W
                                    wr = wd + ky * k
                                    for kx in range(k):
                                        xx = x0 + kx
                                        if xx < 0 or xx >= W:
                                            continue
                                        acc += x[xr + xx] * w[wr + kx]
                        y[orow + ox] = acc

        self.cache = (x, D, H, W)
        return y

    def backward(self, gy):
        x, D, H, W = self.cache
        ic, oc, k, p = self.ic, self.oc, self.k, self.pad
        w = self.w
        gw, gb = self.gw, self.gb
        HW = H * W
        DHW = D * HW
        kkk = k * k * k
        kk = k * k
        ickkk = ic * kkk

        gx = [0.0] * (ic * DHW)
        for o in range(oc):
            wo = o * ickkk
            ob = o * DHW
            gsum = 0.0
            for od in range(D):
                d0 = od - p
                for oy in range(H):
                    y0 = oy - p
                    orow = ob + od * HW + oy * W
                    for ox in range(W):
                        g = gy[orow + ox]
                        if g == 0.0:
                            continue
                        gsum += g
                        x0 = ox - p
                        for c in range(ic):
                            xb = c * DHW
                            wc = wo + c * kkk
                            for kd in range(k):
                                dd = d0 + kd
                                if dd < 0 or dd >= D:
                                    continue
                                xd = xb + dd * HW
                                wd = wc + kd * kk
                                for ky in range(k):
                                    yy = y0 + ky
                                    if yy < 0 or yy >= H:
                                        continue
                                    xr = xd + yy * W
                                    wr = wd + ky * k
                                    for kx in range(k):
                                        xx = x0 + kx
                                        if xx < 0 or xx >= W:
                                            continue
                                        gw[wr + kx] += x[xr + xx] * g
                                        gx[xr + xx] += w[wr + kx] * g
            gb[o] += gsum
        return gx

    def params(self):
        return [(self.w, self.gw), (self.b, self.gb)]


class LeakyReLU:
    """Leaky rectifier. StereoNet uses leaky ReLU throughout."""

    def __init__(self, slope=0.2):
        self.slope = slope
        self.cache = None

    def forward(self, x):
        s = self.slope
        y = [0.0] * len(x)
        for i in range(len(x)):
            v = x[i]
            y[i] = v if v > 0.0 else v * s
        self.cache = x
        return y

    def backward(self, gy):
        x = self.cache
        s = self.slope
        gx = [0.0] * len(gy)
        for i in range(len(gy)):
            gx[i] = gy[i] if x[i] > 0.0 else gy[i] * s
        return gx

    def params(self):
        return []


def zero_grads(layers):
    for layer in layers:
        for _, g in layer.params():
            for i in range(len(g)):
                g[i] = 0.0


# =====================================================================
#  SECTION 6 : STEREO PRIMITIVES
#  Cost volume, differentiable soft-argmin, bilinear resampling.
#  These three are what separate a stereo network from a generic CNN,
#  and all three need hand-written gradients.
# =====================================================================

class CostVolume:
    """Build a matching cost volume by shifting the right feature map.

    The paper concatenates left and right features, giving 2C channels.
    We take the *absolute* difference instead, giving C channels. This
    halves the 3D convolution cost, which is the dominant expense in
    pure Python, and it is standard practice (MC-CNN and descendants).

    The absolute value is not cosmetic. With a signed difference the
    correct match sits at zero but the cost is antisymmetric about it,
    and a stack of convolutions with leaky ReLU is very nearly linear,
    so it cannot produce the V-shaped response soft-argmin needs. With the
    absolute value a good match is small and a bad match is large before
    any learning happens, and the filter only has to sharpen a signal
    that is already pointing the right way.

    Positions where x - d falls outside the right image carry no
    evidence. They are filled with INVALID_COST rather than zero,
    because zero would look like a *perfect* match and would drag the
    soft-argmin toward large disparities all along the left border.

    Input  : fl, fr each [c][y][x], shape (C, H, W)
    Output : volume [c][d][y][x], shape (C, D, H, W)
    """

    INVALID_COST = 2.0

    def __init__(self, D):
        self.D = D
        self.cache = None

    def forward(self, fl, fr, C, H, W):
        D = self.D
        HW = H * W
        DHW = D * HW
        vol = [0.0] * (C * DHW)
        sign = [0.0] * (C * DHW)
        inv = CostVolume.INVALID_COST
        for c in range(C):
            cb = c * HW
            vb = c * DHW
            for d in range(D):
                db = vb + d * HW
                for y in range(H):
                    row = y * W
                    for x in range(W):
                        idx = db + row + x
                        xr = x - d
                        if xr < 0:
                            vol[idx] = inv
                            continue
                        v = fl[cb + row + x] - fr[cb + row + xr]
                        if v < 0.0:
                            vol[idx] = -v
                            sign[idx] = -1.0
                        else:
                            vol[idx] = v
                            sign[idx] = 1.0
        self.cache = (C, H, W, sign)
        return vol

    def backward(self, gvol):
        C, H, W, sign = self.cache
        D = self.D
        HW = H * W
        DHW = D * HW
        gfl = [0.0] * (C * HW)
        gfr = [0.0] * (C * HW)
        for c in range(C):
            cb = c * HW
            vb = c * DHW
            for d in range(D):
                db = vb + d * HW
                for y in range(H):
                    row = y * W
                    for x in range(W):
                        xr = x - d
                        if xr < 0:
                            continue
                        idx = db + row + x
                        g = gvol[idx] * sign[idx]
                        if g == 0.0:
                            continue
                        gfl[cb + row + x] += g
                        gfr[cb + row + xr] -= g
        return gfl, gfr

    def invalid_mask(self, H, W, penalty=8.0):
        """Additive penalty on the *filtered* cost, so a disparity that
        reaches outside the right image can never win the soft-argmin
        whatever the convolutions produce."""
        D = self.D
        HW = H * W
        m = [0.0] * (D * HW)
        for d in range(1, D):
            db = d * HW
            for y in range(H):
                row = y * W
                for x in range(d):
                    m[db + row + x] = penalty
        return m


class SoftArgmin:
    """Differentiable disparity regression (Kendall et al., GC-Net).

        p_d    = softmax_d(-cost_d)
        disp   = sum_d  d * p_d

    A hard argmin would give no gradient and integer-only disparity.
    Soft-argmin gives both a gradient and sub-pixel precision, which is
    exactly what the 3D reconstruction downstream needs.
    """

    def __init__(self, D):
        self.D = D
        self.cache = None

    def forward(self, cost, H, W, penalty=None):
        D = self.D
        HW = H * W
        if penalty is not None:
            cost = [cost[i] + penalty[i] for i in range(len(cost))]
        disp = [0.0] * HW
        probs = [0.0] * (D * HW)
        for i in range(HW):
            # Numerically stable softmax over the disparity axis.
            best = -cost[i]
            for d in range(1, D):
                v = -cost[d * HW + i]
                if v > best:
                    best = v
            total = 0.0
            for d in range(D):
                e = fexp(-cost[d * HW + i] - best)
                probs[d * HW + i] = e
                total += e
            inv = 1.0 / total
            acc = 0.0
            for d in range(D):
                p = probs[d * HW + i] * inv
                probs[d * HW + i] = p
                acc += d * p
            disp[i] = acc
        self.cache = (probs, disp, H, W)
        return disp

    def backward(self, gdisp):
        probs, disp, H, W = self.cache
        D = self.D
        HW = H * W
        gcost = [0.0] * (D * HW)
        for i in range(HW):
            g = gdisp[i]
            if g == 0.0:
                continue
            di = disp[i]
            for d in range(D):
                idx = d * HW + i
                # d(disp)/d(cost_d) = p_d * (disp - d)
                gcost[idx] = g * probs[idx] * (di - d)
        return gcost


class BilinearUpsample:
    """Bilinear resampling, used to climb the refinement pyramid.

    Half-pixel centre convention: src = (dst + 0.5) * scale - 0.5.
    """

    def __init__(self):
        self.cache = None

    def forward(self, x, C, H, W, H2, W2):
        sy = H / float(H2)
        sx = W / float(W2)
        out = [0.0] * (C * H2 * W2)
        taps = []
        HW = H * W
        for oy in range(H2):
            fy = (oy + 0.5) * sy - 0.5
            if fy < 0.0:
                fy = 0.0
            y0 = int(fy)
            if y0 > H - 1:
                y0 = H - 1
            y1 = y0 + 1 if y0 + 1 < H else y0
            wy = fy - y0
            for ox in range(W2):
                fx = (ox + 0.5) * sx - 0.5
                if fx < 0.0:
                    fx = 0.0
                x0 = int(fx)
                if x0 > W - 1:
                    x0 = W - 1
                x1 = x0 + 1 if x0 + 1 < W else x0
                wx = fx - x0
                w00 = (1.0 - wy) * (1.0 - wx)
                w01 = (1.0 - wy) * wx
                w10 = wy * (1.0 - wx)
                w11 = wy * wx
                i00 = y0 * W + x0
                i01 = y0 * W + x1
                i10 = y1 * W + x0
                i11 = y1 * W + x1
                taps.append((oy * W2 + ox, i00, w00, i01, w01,
                             i10, w10, i11, w11))

        for c in range(C):
            cb = c * HW
            ob = c * H2 * W2
            for (od, i00, w00, i01, w01, i10, w10, i11, w11) in taps:
                out[ob + od] = (x[cb + i00] * w00 + x[cb + i01] * w01 +
                                x[cb + i10] * w10 + x[cb + i11] * w11)

        self.cache = (taps, C, H, W, H2, W2)
        return out

    def backward(self, gy):
        taps, C, H, W, H2, W2 = self.cache
        HW = H * W
        gx = [0.0] * (C * HW)
        for c in range(C):
            cb = c * HW
            ob = c * H2 * W2
            for (od, i00, w00, i01, w01, i10, w10, i11, w11) in taps:
                g = gy[ob + od]
                if g == 0.0:
                    continue
                gx[cb + i00] += g * w00
                gx[cb + i01] += g * w01
                gx[cb + i10] += g * w10
                gx[cb + i11] += g * w11
        return gx

    def params(self):
        return []


def avgpool2(img, C, H, W):
    """2x2 average pooling, for building the image pyramid."""
    H2 = H // 2
    W2 = W // 2
    out = [0.0] * (C * H2 * W2)
    for c in range(C):
        cb = c * H * W
        ob = c * H2 * W2
        for y in range(H2):
            r0 = cb + (2 * y) * W
            r1 = r0 + W
            orow = ob + y * W2
            for x in range(W2):
                x0 = 2 * x
                out[orow + x] = 0.25 * (img[r0 + x0] + img[r0 + x0 + 1] +
                                        img[r1 + x0] + img[r1 + x0 + 1])
    return out, H2, W2


# =====================================================================
#  SECTION 7 : THE NETWORK
#
#  StereoNet, scaled for pure-Python compute:
#
#    left  ---> [siamese feature net, /4] ---\
#                                             >-- cost volume (C,D,h,w)
#    right ---> [siamese feature net, /4] ---/
#                          |
#                   3x 3D convolution
#                          |
#                    soft-argmin  ->  coarse disparity at 1/4
#                          |
#          upsample x2 + edge-aware refinement (guided by image at 1/2)
#                          |
#          upsample x2 + edge-aware refinement (guided by image at 1/1)
#                          |
#                   full-resolution disparity
#
#  The siamese branch shares one set of weights across both images, so
#  forward is run twice and the layer caches are stashed and restored
#  around the two backward passes.
# =====================================================================

class FeatureNet:
    """Shared-weight feature extractor. Downsamples 4x."""

    def __init__(self, rng, C=8):
        self.c1 = Conv2d(1, C, 3, rng, stride=2)
        self.a1 = LeakyReLU()
        self.c2 = Conv2d(C, C, 3, rng, stride=2)
        self.a2 = LeakyReLU()
        self.c3 = Conv2d(C, C, 3, rng, stride=1)
        self.a3 = LeakyReLU()
        self.c4 = Conv2d(C, C, 3, rng, stride=1)
        self.C = C
        self.layers = [self.c1, self.a1, self.c2, self.a2,
                       self.c3, self.a3, self.c4]

    def forward(self, img, H, W):
        x, h, w = self.c1.forward(img, H, W)
        x = self.a1.forward(x)
        x, h, w = self.c2.forward(x, h, w)
        x = self.a2.forward(x)
        x, h, w = self.c3.forward(x, h, w)
        x = self.a3.forward(x)
        x, h, w = self.c4.forward(x, h, w)
        return x, h, w

    def backward(self, gx):
        gx = self.c4.backward(gx)
        gx = self.a3.backward(gx)
        gx = self.c3.backward(gx)
        gx = self.a2.backward(gx)
        gx = self.c2.backward(gx)
        gx = self.a1.backward(gx)
        gx = self.c1.backward(gx)
        return gx

    def save_cache(self):
        return [l.cache for l in self.layers]

    def load_cache(self, tape):
        for l, c in zip(self.layers, tape):
            l.cache = c

    def params(self):
        out = []
        for l in self.layers:
            out += l.params()
        return out


class CostFilter:
    """Three 3D convolutions that regularise the raw matching cost."""

    def __init__(self, rng, Cin=8, Cmid=4):
        self.c1 = Conv3d(Cin, Cmid, 3, rng)
        self.a1 = LeakyReLU()
        self.c2 = Conv3d(Cmid, Cmid, 3, rng)
        self.a2 = LeakyReLU()
        self.c3 = Conv3d(Cmid, 1, 3, rng)
        # Start the final layer small. A large initial cost spread makes
        # the softmax nearly one-hot from the first iteration, and a
        # saturated soft-argmin has no gradient to recover with.
        for i in range(len(self.c3.w)):
            self.c3.w[i] *= 0.25
        self.layers = [self.c1, self.a1, self.c2, self.a2, self.c3]

    def forward(self, vol, D, H, W):
        x = self.c1.forward(vol, D, H, W)
        x = self.a1.forward(x)
        x = self.c2.forward(x, D, H, W)
        x = self.a2.forward(x)
        x = self.c3.forward(x, D, H, W)
        return x

    def backward(self, g):
        g = self.c3.backward(g)
        g = self.a2.backward(g)
        g = self.c2.backward(g)
        g = self.a1.backward(g)
        g = self.c1.backward(g)
        return g

    def params(self):
        out = []
        for l in self.layers:
            out += l.params()
        return out


class Refinement:
    """Edge-aware hierarchical refinement, one pyramid level.

    Takes the upsampled disparity and the guide image at this level,
    concatenated as two channels, and predicts a residual. Dilation
    widens the receptive field without extra cost. This is the mechanism the
    paper relies on to recover thin structures and sharp depth edges
    that the 1/4-resolution cost volume cannot represent.
    """

    def __init__(self, rng, C=8, dilations=(1, 2, 4)):
        self.dilations = tuple(dilations)
        self.convs = []
        self.acts = []
        prev = 2
        for d in self.dilations:
            self.convs.append(Conv2d(prev, C, 3, rng, stride=1, dil=d))
            self.acts.append(LeakyReLU())
            prev = C
        self.c3 = Conv2d(prev, 1, 3, rng, stride=1, dil=1)
        self.out_relu = LeakyReLU(0.0)      # disparity cannot be negative
        self.layers = []
        for i in range(len(self.convs)):
            self.layers.append(self.convs[i])
            self.layers.append(self.acts[i])
        self.layers.append(self.c3)
        self.layers.append(self.out_relu)

    def forward(self, disp, guide, H, W):
        HW = H * W
        x = [0.0] * (2 * HW)
        for i in range(HW):
            x[i] = disp[i]
            x[HW + i] = guide[i]
        self.HW = HW
        y = x
        for i in range(len(self.convs)):
            y, _, _ = self.convs[i].forward(y, H, W)
            y = self.acts[i].forward(y)
        res, _, _ = self.c3.forward(y, H, W)
        summed = [disp[i] + res[i] for i in range(HW)]
        return self.out_relu.forward(summed)

    def backward(self, gout):
        HW = self.HW
        g = self.out_relu.backward(gout)     # d/d(disp + res)
        gres = self.c3.backward(g)
        for i in range(len(self.convs) - 1, -1, -1):
            gres = self.acts[i].backward(gres)
            gres = self.convs[i].backward(gres)
        gin = gres
        # The skip connection adds a second path back to disp.
        gdisp = [gin[i] + g[i] for i in range(HW)]
        return gdisp

    def params(self):
        out = []
        for l in self.layers:
            out += l.params()
        return out


class StereoNet:
    """The full model."""

    def __init__(self, rng, feat_C=8, cost_C=4, D=8, max_disp=32,
                 refine_C=8, refine_dilations=(1, 2, 4)):
        self.feat = FeatureNet(rng, feat_C)
        self.cv = CostVolume(D)
        self.filt = CostFilter(rng, feat_C, cost_C)
        self.argmin = SoftArgmin(D)
        self.up1 = BilinearUpsample()
        self.up2 = BilinearUpsample()
        self.ref1 = Refinement(rng, refine_C, refine_dilations)
        self.ref2 = Refinement(rng, refine_C, refine_dilations)
        self.D = D
        self.feat_C = feat_C
        self.max_disp = max_disp
        # Operating point for scale selection: the peak disparity, in
        # working-resolution pixels, that predict_auto aims for. It
        # belongs to the trained weights, not to the code, because it is
        # tuned on validation separately for each model -- 20 for the
        # D=8 network, higher once the search is widened.
        self.target_peak = 20

    def all_params(self):
        return (self.feat.params() + self.filt.params() +
                self.ref1.params() + self.ref2.params())

    def modules(self):
        return (self.feat.layers + self.filt.layers +
                self.ref1.layers + self.ref2.layers)

    def forward(self, left, right, H, W, guide_half, guide_full):
        """Returns (disp_quarter, disp_half, disp_full) in *pixels at
        their own resolution*."""
        # --- siamese features, weights shared ---
        fl, h, w = self.feat.forward(left, H, W)
        self.tape_l = self.feat.save_cache()
        fr, _, _ = self.feat.forward(right, H, W)
        self.tape_r = self.feat.save_cache()
        self.h, self.w = h, w

        # --- cost volume and filtering ---
        vol = self.cv.forward(fl, fr, self.feat_C, h, w)
        cost = self.filt.forward(vol, self.D, h, w)

        # --- coarse disparity at 1/4 resolution ---
        d0 = self.argmin.forward(cost, h, w,
                                 self.cv.invalid_mask(h, w))

        # --- refinement level 1 : 1/4 -> 1/2 ---
        H2, W2 = H // 2, W // 2
        d0_up = self.up1.forward(d0, 1, h, w, H2, W2)
        d0_up = [v * 2.0 for v in d0_up]     # disparity scales with res
        d1 = self.ref1.forward(d0_up, guide_half, H2, W2)

        # --- refinement level 2 : 1/2 -> 1/1 ---
        d1_up = self.up2.forward(d1, 1, H2, W2, H, W)
        d1_up = [v * 2.0 for v in d1_up]
        d2 = self.ref2.forward(d1_up, guide_full, H, W)

        self.shapes = (H, W, H2, W2, h, w)
        return d0, d1, d2

    def backward(self, g0, g1, g2, gcost_extra=None):
        """g0, g1, g2 are gradients on the three outputs. gcost_extra is
        an additional gradient applied directly to the filtered cost,
        used by the distribution cross-entropy."""
        H, W, H2, W2, h, w = self.shapes

        # level 2
        gd1_up = self.ref2.backward(g2)
        gd1_up = [v * 2.0 for v in gd1_up]
        gd1 = self.up2.backward(gd1_up)
        for i in range(len(gd1)):
            gd1[i] += g1[i]

        # level 1
        gd0_up = self.ref1.backward(gd1)
        gd0_up = [v * 2.0 for v in gd0_up]
        gd0 = self.up1.backward(gd0_up)
        for i in range(len(gd0)):
            gd0[i] += g0[i]

        # cost volume path
        gcost = self.argmin.backward(gd0)
        if gcost_extra is not None:
            for i in range(len(gcost)):
                gcost[i] += gcost_extra[i]
        gvol = self.filt.backward(gcost)
        gfl, gfr = self.cv.backward(gvol)

        # siamese: accumulate through both branches into shared weights
        self.feat.load_cache(self.tape_r)
        self.feat.backward(gfr)
        self.feat.load_cache(self.tape_l)
        self.feat.backward(gfl)


# =====================================================================
#  SECTION 8 : LOSS AND OPTIMISER
# =====================================================================

def disparity_ce(probs, target, valid, D, H, W, sigma=0.8):
    """Cross-entropy between the predicted disparity distribution and a
    Gaussian bump centred on the ground-truth disparity.

    Regression through soft-argmin has a fatal failure mode: once the
    softmax saturates, its gradient p_k*(disp - k) collapses toward zero,
    so a model confidently wrong about the disparity has almost no
    gradient with which to change its mind. Cross-entropy does not share
    that property. With p = softmax(-cost),

        dL/d(cost_k) = q_k - p_k

    which is *largest* exactly when the prediction is confident and
    wrong. Supervising the distribution directly, alongside the
    regression loss on the final disparity, is what lets the cost volume
    learn to match at all.
    """
    HW = H * W
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    gcost = [0.0] * (D * HW)
    total = 0.0
    count = 0
    for i in range(HW):
        if not valid[i]:
            continue
        dg = target[i]
        # Gaussian target distribution over the disparity axis.
        q = [0.0] * D
        qsum = 0.0
        for d in range(D):
            e = d - dg
            qv = fexp(-e * e * inv2s2)
            q[d] = qv
            qsum += qv
        if qsum <= 0.0:
            continue
        qinv = 1.0 / qsum
        for d in range(D):
            q[d] *= qinv
            pk = probs[d * HW + i]
            if pk > 1.0e-12:
                total -= q[d] * flog(pk)
            gcost[d * HW + i] = q[d] - pk
        count += 1
    if count == 0:
        return 0.0, gcost, 0
    inv = 1.0 / count
    for i in range(len(gcost)):
        gcost[i] *= inv
    return total * inv, gcost, count


def pseudo_huber(pred, target, valid, alpha=1.0):
    """Robust loss  rho(x) = sqrt(x^2 + a^2) - a.

    Quadratic near zero for a clean gradient on easy pixels, linear far
    out so occlusions and mismatches cannot dominate. This is the role
    the two-parameter robust loss plays in the paper.
    """
    n = len(pred)
    a2 = alpha * alpha
    total = 0.0
    grad = [0.0] * n
    count = 0
    for i in range(n):
        if valid is not None and not valid[i]:
            continue
        x = pred[i] - target[i]
        r = fsqrt(x * x + a2)
        total += r - alpha
        grad[i] = x / r
        count += 1
    if count == 0:
        return 0.0, grad, 0
    inv = 1.0 / count
    for i in range(n):
        grad[i] *= inv
    return total * inv, grad, count


class Adam:
    """Adam. Bias correction uses running products, not pow()."""

    def __init__(self, params, lr=1.0e-3, b1=0.9, b2=0.999, eps=1.0e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [[0.0] * len(p) for p, _ in params]
        self.v = [[0.0] * len(p) for p, _ in params]
        self.b1t = 1.0
        self.b2t = 1.0

    def step(self, clip=5.0):
        self.b1t *= self.b1
        self.b2t *= self.b2
        c1 = 1.0 / (1.0 - self.b1t)
        c2 = 1.0 / (1.0 - self.b2t)
        b1, b2, eps, lr = self.b1, self.b2, self.eps, self.lr
        for k in range(len(self.params)):
            p, g = self.params[k]
            m, v = self.m[k], self.v[k]
            for i in range(len(p)):
                gi = g[i]
                if gi > clip:
                    gi = clip
                elif gi < -clip:
                    gi = -clip
                mi = b1 * m[i] + (1.0 - b1) * gi
                vi = b2 * v[i] + (1.0 - b2) * gi * gi
                m[i] = mi
                v[i] = vi
                p[i] -= lr * (mi * c1) / (fsqrt(vi * c2) + eps)


# =====================================================================
#  SECTION 9 : SYNTHETIC DATA: RANDOM-DOT STEREOGRAMS
#
#  Random-dot stereograms (Julesz, 1971) are the canonical stereo test
#  case: they carry no monocular depth cue whatsoever, so a model that
#  solves them must be doing genuine binocular matching rather than
#  guessing from shading or familiar object size. They also give exact
#  ground-truth disparity for free and require no dataset download and
#  no image decoding, which matters when PIL is off the table.
#
#  Each scene is a slanted background plane plus a few fronto-parallel
#  and slanted boxes, textured with band-limited noise and given flat
#  colours so the reconstruction stage downstream has something to
#  paint onto the Gaussians.
# =====================================================================

def _box_blur(img, W, H, radius=1):
    """Separable box blur. Makes the dot texture band-limited so that
    matching is well posed rather than aliased."""
    tmp = [0.0] * (W * H)
    out = [0.0] * (W * H)
    for y in range(H):
        row = y * W
        for x in range(W):
            acc = 0.0
            cnt = 0
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if 0 <= xx < W:
                    acc += img[row + xx]
                    cnt += 1
            tmp[row + x] = acc / cnt
    for y in range(H):
        for x in range(W):
            acc = 0.0
            cnt = 0
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if 0 <= yy < H:
                    acc += tmp[yy * W + x]
                    cnt += 1
            out[y * W + x] = acc / cnt
    return out


def make_stereogram(rng, W, H, max_disp=16, n_objects=3, colour=True):
    """Generate one stereo pair with exact ground-truth disparity.

    Returns dict with keys:
        left, right   greyscale, flat W*H, values in [0,1]
        disp          ground-truth disparity of the LEFT image, pixels
        left_rgb      colour left image, flat 3*W*H
        valid         per-pixel validity (False where occluded/unseen)
    """
    HW = W * H

    # ---- 1. disparity map (a depth scene, expressed as disparity) ----
    disp = [0.0] * HW
    label = [0] * HW            # object id per pixel, for colouring

    # Background: a plane slanted in depth, so it is not trivially flat.
    # Same reasoning as make_textured_scene: keep the depth ratio
    # realistic rather than letting the background recede to infinity.
    back_far = rng.range(0.30, 0.45) * max_disp
    back_near = back_far * rng.range(1.15, 1.50)
    for y in range(H):
        t = y / float(H - 1)
        d = back_near + (back_far - back_near) * t
        row = y * W
        for x in range(W):
            disp[row + x] = d

    # Foreground objects at larger disparity (nearer the camera).
    for obj in range(n_objects):
        oid = obj + 1
        ow = rng.randint(W // 6, W // 2)
        oh = rng.randint(H // 6, H // 2)
        ox = rng.randint(0, W - ow - 1)
        oy = rng.randint(0, H - oh - 1)
        d_base = rng.range(max_disp * 0.58, max_disp * 0.95)
        slant_x = rng.range(-1.5, 1.5) / fmax(1.0, ow)
        slant_y = rng.range(-1.5, 1.5) / fmax(1.0, oh)
        ellipse = rng.uniform() < 0.45
        cx = ox + ow * 0.5
        cy = oy + oh * 0.5
        rx = ow * 0.5
        ry = oh * 0.5
        for y in range(oy, oy + oh):
            row = y * W
            for x in range(ox, ox + ow):
                if ellipse:
                    nx = (x - cx) / rx
                    ny = (y - cy) / ry
                    if nx * nx + ny * ny > 1.0:
                        continue
                d = d_base + slant_x * (x - ox) + slant_y * (y - oy)
                if d > max_disp:
                    d = float(max_disp)
                if d < 0.0:
                    d = 0.0
                disp[row + x] = d
                label[row + x] = oid

    # ---- 2. band-limited random texture, wide enough to shift into ----
    TW = W + max_disp + 4
    raw = [rng.uniform() for _ in range(TW * H)]
    tex = _box_blur(raw, TW, H, 1)
    # Stretch contrast so the texture uses the full range.
    tlo = min(tex)
    thi = max(tex)
    tspan = thi - tlo if thi > tlo else 1.0
    for i in range(len(tex)):
        tex[i] = (tex[i] - tlo) / tspan

    # ---- 3. left image is the texture, right is the warped texture ----
    left = [0.0] * HW
    for y in range(H):
        srow = y * TW
        drow = y * W
        for x in range(W):
            left[drow + x] = tex[srow + x + max_disp]

    # Scatter left pixels into the right image at x - d, with a z-buffer
    # so nearer surfaces (larger disparity) correctly occlude farther
    # ones. This is what produces realistic occlusion boundaries.
    right = [-1.0] * HW
    zbuf = [-1.0] * HW
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            xr = int(x - d + 0.5)
            if xr < 0 or xr >= W:
                continue
            if d > zbuf[row + xr]:
                zbuf[row + xr] = d
                right[row + xr] = left[row + x]

    # Disoccluded pixels in the right view were never seen on the left;
    # fill them with fresh texture, which is what a real camera records.
    for i in range(HW):
        if right[i] < 0.0:
            y = i // W
            x = i % W
            right[i] = tex[y * TW + x]

    # ---- 4. validity mask: a left pixel is valid if it is visible on
    #         the right (not occluded and not off the image edge) ----
    valid = [False] * HW
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            xr = int(x - d + 0.5)
            if 0 <= xr < W and zbuf[row + xr] <= d + 1.0e-6:
                valid[row + x] = True

    # ---- 5. colour, so the Gaussians have something to show ----
    left_rgb = None
    if colour:
        left_rgb = [0.0] * (3 * HW)
        palette = [(0.55, 0.58, 0.62)]
        for _ in range(n_objects):
            palette.append((rng.range(0.25, 1.0),
                            rng.range(0.25, 1.0),
                            rng.range(0.25, 1.0)))
        for i in range(HW):
            r, g, b = palette[label[i]]
            t = 0.45 + 0.75 * left[i]       # texture modulates the hue
            left_rgb[3 * i] = fclamp(r * t, 0.0, 1.0)
            left_rgb[3 * i + 1] = fclamp(g * t, 0.0, 1.0)
            left_rgb[3 * i + 2] = fclamp(b * t, 0.0, 1.0)

    return {'left': left, 'right': right, 'disp': disp,
            'left_rgb': left_rgb, 'valid': valid, 'W': W, 'H': H,
            'max_disp': max_disp}


# =====================================================================
#  SECTION 10 : TRAINING AND EVALUATION
# =====================================================================

def pool_disp(disp, W, H):
    """Halve a disparity map's resolution. Disparity is measured in
    pixels, so the values must be halved along with the grid."""
    out, H2, W2 = avgpool2(disp, 1, H, W)
    return [v * 0.5 for v in out], W2, H2


def pool_valid(valid, W, H):
    """Halve a validity mask. A coarse pixel is valid only if all four
    of its children are (conservative on purpose)."""
    H2 = H // 2
    W2 = W // 2
    out = [False] * (W2 * H2)
    for y in range(H2):
        r0 = (2 * y) * W
        r1 = r0 + W
        for x in range(W2):
            x0 = 2 * x
            out[y * W2 + x] = (valid[r0 + x0] and valid[r0 + x0 + 1] and
                               valid[r1 + x0] and valid[r1 + x0 + 1])
    return out, W2, H2


def build_targets(sample):
    """Ground truth and guide images at all three pyramid levels."""
    W, H = sample['W'], sample['H']
    d_full = sample['disp']
    v_full = sample['valid']
    g_full = sample['left']

    d_half, W2, H2 = pool_disp(d_full, W, H)
    v_half, _, _ = pool_valid(v_full, W, H)
    g_half, _, _ = avgpool2(g_full, 1, H, W)

    d_qtr, W4, H4 = pool_disp(d_half, W2, H2)
    v_qtr, _, _ = pool_valid(v_half, W2, H2)

    return {'d': [d_qtr, d_half, d_full],
            'v': [v_qtr, v_half, v_full],
            'guide_half': g_half, 'guide_full': g_full,
            'dims': [(W4, H4), (W2, H2), (W, H)]}


LEVEL_WEIGHTS = [0.5, 0.7, 1.0]     # coarse to fine, as in the paper
CE_WEIGHT = 2.0                     # distribution supervision on the cost volume


def train_grads(net, sample):
    """Forward and backward on one sample, leaving gradients in the
    parameters and returning the loss. Split out of train_step so that
    several processes can each compute gradients on their own sample and
    have them averaged before a single optimiser step -- batch training
    with no change to the model code."""
    tgt = build_targets(sample)
    W, H = sample['W'], sample['H']

    zero_grads(net.modules())
    preds = net.forward(sample['left'], sample['right'], H, W,
                        tgt['guide_half'], tgt['guide_full'])

    total = 0.0
    grads = []
    for lvl in range(3):
        loss, g, n = pseudo_huber(preds[lvl], tgt['d'][lvl], tgt['v'][lvl])
        wgt = LEVEL_WEIGHTS[lvl]
        total += wgt * loss
        grads.append([v * wgt for v in g])

    # Distribution supervision on the cost volume.
    gcost_extra = None
    if CE_WEIGHT > 0.0:
        probs, _, hh, ww = net.argmin.cache
        W4, H4 = tgt['dims'][0]
        ce, gce, n = disparity_ce(probs, tgt['d'][0], tgt['v'][0],
                                  net.D, H4, W4)
        total += CE_WEIGHT * ce
        # The helper returns q - p, which IS dL/d(cost): the softmax is
        # taken over -cost, so the two sign flips cancel.
        gcost_extra = [CE_WEIGHT * v for v in gce]

    net.backward(grads[0], grads[1], grads[2], gcost_extra)
    return total, preds[2], tgt


def train_step(net, opt, sample):
    total, pred, tgt = train_grads(net, sample)
    opt.step()
    return total, pred, tgt


def flat_params(net):
    out = []
    for p, _ in net.all_params():
        out.extend(p)
    return out


def set_flat_params(net, flat):
    k = 0
    for p, _ in net.all_params():
        for i in range(len(p)):
            p[i] = flat[k]
            k += 1


def flat_grads(net):
    out = []
    for _, g in net.all_params():
        out.extend(g)
    return out


def set_flat_grads(net, flat):
    k = 0
    for _, g in net.all_params():
        for i in range(len(g)):
            g[i] = flat[k]
            k += 1


def metrics(pred, target, valid):
    """End-point error and bad-pixel rates, the standard stereo triple."""
    epe = 0.0
    bad1 = 0
    bad3 = 0
    n = 0
    for i in range(len(pred)):
        if not valid[i]:
            continue
        e = fabs(pred[i] - target[i])
        epe += e
        if e > 1.0:
            bad1 += 1
        if e > 3.0:
            bad3 += 1
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return epe / n, 100.0 * bad1 / n, 100.0 * bad3 / n


def diagnose(net, rng, n_scenes=5, W=64, H=64, max_disp=16, sampler=None):
    """Per-level evaluation against the trivial baseline.

    EPE alone will happily report a respectable number for a model that
    has learned to output the average disparity and nothing else. The
    only defence is to compare every level against the cost of simply
    predicting the mean, and to look at the output as an image. An
    earlier version of this network was *worse than the constant* at
    both coarse levels while reporting EPE 2.379 px, which is how this
    function came to exist.
    """
    names = ['1/4  cost volume + soft-argmin',
             '1/2  after refinement 1',
             '1/1  after refinement 2']
    tot = [0.0, 0.0, 0.0]
    base = [0.0, 0.0, 0.0]
    for k in range(n_scenes):
        if sampler is None:
            s = make_stereogram(rng, W, H, max_disp=max_disp,
                                n_objects=rng.randint(2, 4))
        else:
            s = sampler(rng, W, H, max_disp)
        tgt = build_targets(s)
        preds = net.forward(s['left'], s['right'], H, W,
                            tgt['guide_half'], tgt['guide_full'])
        for lvl in range(3):
            d = tgt['d'][lvl]
            v = tgt['v'][lvl]
            e, _, _ = metrics(preds[lvl], d, v)
            tot[lvl] += e
            nv = 0
            acc = 0.0
            for i in range(len(d)):
                if v[i]:
                    acc += d[i]
                    nv += 1
            mean = acc / fmax(1, nv)
            eb, _, _ = metrics([mean] * len(d), d, v)
            base[lvl] += eb

    print('level                              model EPE   constant EPE   verdict')
    ok = True
    for lvl in range(3):
        m = tot[lvl] / n_scenes
        b = base[lvl] / n_scenes
        good = m < b
        if not good:
            ok = False
        print('  %-32s %7.3f      %7.3f      %s'
              % (names[lvl], m, b, 'ok' if good else 'WORSE THAN CONSTANT'))
    return ok


def save_weights(net, path):
    """Plain-text checkpoint. No pickle, since pickle is an import.

    The first line records the architecture. Without it a checkpoint is
    just a pile of numbers whose shape has to be guessed, and changing
    the model silently breaks every saved file. That
    happened here once the refinement stage was widened.
    """
    f = open(path, 'w')
    params = net.all_params()
    f.write('stereonet feat_C=%d cost_C=%d D=%d max_disp=%d refine_C=%d '
            'refine_dil=%s target_peak=%d\n'
            % (net.feat_C, net.filt.c1.oc, net.D, net.max_disp,
               net.ref1.convs[0].oc,
               ','.join([str(d) for d in net.ref1.dilations]),
               int(getattr(net, 'target_peak', 20))))
    f.write('%d\n' % len(params))
    for p, _ in params:
        f.write('%d ' % len(p))
        f.write(' '.join(['%.9g' % v for v in p]))
        f.write('\n')
    f.close()


def read_checkpoint_config(path):
    """Read the architecture header, or None for a headerless file
    written before the header existed."""
    f = open(path, 'r')
    first = f.readline().strip()
    f.close()
    if not first.startswith('stereonet '):
        return None
    cfg = {}
    for tok in first.split()[1:]:
        if '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        if k == 'refine_dil':
            cfg[k] = tuple([int(d) for d in v.split(',')])
        else:
            cfg[k] = int(v)
    return cfg


def load_model(path, rng=None):
    """Build a StereoNet matching a checkpoint, then load it."""
    cfg = read_checkpoint_config(path)
    if rng is None:
        rng = Rng(1)
    if cfg is not None:
        net = StereoNet(rng, feat_C=cfg.get('feat_C', 8),
                        cost_C=cfg.get('cost_C', 4),
                        D=cfg.get('D', 8),
                        max_disp=cfg.get('max_disp', 32),
                        refine_C=cfg.get('refine_C', 8),
                        refine_dilations=cfg.get('refine_dil', (1, 2, 4)))
        net.target_peak = cfg.get('target_peak', 20)
        load_weights(net, path)
        return net

    # Headerless checkpoint, written before the header existed. Try the
    # architectures this project has actually shipped and keep the one
    # whose parameter block matches the file, so an old checkpoint keeps
    # working instead of failing with a shape error.
    f = open(path, 'r')
    n_blocks = int(f.readline().strip())
    sizes = []
    for _ in range(n_blocks):
        line = f.readline()
        sizes.append(int(line.split(' ', 1)[0]))
    f.close()

    candidates = [(8, 4, 4, (1, 2)), (8, 4, 8, (1, 2, 4)),
                  (12, 6, 8, (1, 2, 4)), (16, 8, 8, (1, 2, 4))]
    for (fc, cc, rc, dil) in candidates:
        trial = StereoNet(Rng(1), feat_C=fc, cost_C=cc, D=8, max_disp=32,
                          refine_C=rc, refine_dilations=dil)
        tp = trial.all_params()
        if len(tp) != n_blocks:
            continue
        if all(len(tp[i][0]) == sizes[i] for i in range(n_blocks)):
            net = StereoNet(rng, feat_C=fc, cost_C=cc, D=8, max_disp=32,
                            refine_C=rc, refine_dilations=dil)
            load_weights(net, path)
            return net
    raise ValueError('checkpoint %s does not match any known '
                     'architecture; it may be from a different build'
                     % path)


def load_weights(net, path):
    f = open(path, 'r')
    lines = f.read().split('\n')
    f.close()
    off = 0
    if lines[0].startswith('stereonet '):
        off = 1
    count = int(lines[off])
    params = net.all_params()
    if count != len(params):
        raise ValueError('checkpoint does not match model')
    for k in range(count):
        parts = lines[k + 1 + off].split()
        n = int(parts[0])
        p, _ = params[k]
        if n != len(p):
            raise ValueError('parameter %d size mismatch' % k)
        for i in range(n):
            p[i] = float(parts[i + 1])


def train(net, rng, iters=300, lr=2.0e-3, W=64, H=64, max_disp=16,
          log_every=10, ckpt='stereonet_weights.txt', log_path=None,
          sampler=None):
    opt = Adam(net.all_params(), lr=lr)
    history = []
    run_loss = 0.0
    for it in range(1, iters + 1):
        # Learning-rate decay in the last third of the run.
        if it == int(iters * 0.66):
            opt.lr = lr * 0.3
        if it == int(iters * 0.88):
            opt.lr = lr * 0.1

        if sampler is None:
            sample = make_stereogram(rng, W, H, max_disp=max_disp,
                                     n_objects=rng.randint(2, 4))
        else:
            sample = sampler(rng, W, H, max_disp)
        loss, pred, tgt = train_step(net, opt, sample)
        run_loss += loss

        if it % log_every == 0:
            epe, b1, b3 = metrics(pred, tgt['d'][2], tgt['v'][2])
            avg = run_loss / log_every
            run_loss = 0.0
            line = ('iter %4d  loss %.4f  EPE %.3f px  bad1.0 %5.1f%%  '
                    'bad3.0 %5.1f%%' % (it, avg, epe, b1, b3))
            print(line)
            history.append((it, avg, epe, b1, b3))
            if log_path is not None:
                f = open(log_path, 'a')
                f.write(line + '\n')
                f.close()
            save_weights(net, ckpt)
    save_weights(net, ckpt)
    return history


def evaluate(net, rng, n_scenes=8, W=64, H=64, max_disp=16, sampler=None):
    """Held-out evaluation on freshly generated scenes."""
    tot_epe = 0.0
    tot_b1 = 0.0
    tot_b3 = 0.0
    for _ in range(n_scenes):
        if sampler is None:
            s = make_stereogram(rng, W, H, max_disp=max_disp,
                                n_objects=rng.randint(2, 4))
        else:
            s = sampler(rng, W, H, max_disp)
        tgt = build_targets(s)
        preds = net.forward(s['left'], s['right'], H, W,
                            tgt['guide_half'], tgt['guide_full'])
        epe, b1, b3 = metrics(preds[2], tgt['d'][2], tgt['v'][2])
        tot_epe += epe
        tot_b1 += b1
        tot_b3 += b3
    return tot_epe / n_scenes, tot_b1 / n_scenes, tot_b3 / n_scenes


# =====================================================================
#  SECTION 11 : GEOMETRY: DISPARITY TO A 3D POINT CLOUD
#
#  For a rectified stereo pair the relationship is exact:
#
#        Z = f * B / (d + doffs)
#        X = (u - cx) * Z / f
#        Y = (v - cy) * Z / f
#
#  f is focal length in pixels, B the baseline, d the disparity, and
#  doffs the principal-point offset between the two cameras (zero for
#  our synthetic rig, non-zero for Middlebury pairs).
# =====================================================================

class Camera:
    """Pinhole intrinsics plus a stereo baseline."""

    def __init__(self, fx, cx, cy, baseline, doffs=0.0):
        self.fx = fx
        self.cx = cx
        self.cy = cy
        self.baseline = baseline
        self.doffs = doffs

    @staticmethod
    def synthetic(W, H, baseline=0.10):
        """Intrinsics for our generated scenes.

        A 60-degree horizontal field of view gives fx = (W/2)/tan(30deg).
        With a 10 cm baseline this puts a disparity of 16 px at roughly
        half a metre and 2 px at about four metres, which is a sensible
        indoor working range.
        """
        fx = (W * 0.5) / 0.5773502691896257      # tan(30 degrees)
        return Camera(fx, W * 0.5, H * 0.5, baseline, 0.0)

    @staticmethod
    def from_calib(path):
        c = read_calib(path)
        return Camera(c['fx'], c['cx'], c['cy'], c['baseline'], c['doffs'])

    def depth_from_disparity(self, d):
        den = d + self.doffs
        if den < 1.0e-6:
            return 0.0
        return self.fx * self.baseline / den

    def unproject(self, u, v, Z):
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fx
        return X, Y, Z


def robust_min_disparity(disp, floor=0.4, factor=0.38):
    """A disparity floor that keeps the depth range sane.

    Depth is f*B/d, so a pixel whose disparity drifts near zero
    unprojects to an effectively infinite distance. One such pixel is
    enough to wreck a reconstruction: it stretches the bounding volume
    so far that every real surface collapses to a speck, and it drags a
    mesh triangle out to the horizon. Clamping at a fraction of the
    median disparity bounds the depth range at roughly 1/factor times
    the median depth, which is generous for a real scene and fatal to
    the outliers.
    """
    vals = sorted(disp)
    med = vals[len(vals) // 2]
    return fmax(floor, med * factor)


def disparity_to_points(disp, W, H, cam, min_disp=0.5, max_depth=1.0e6):
    """Unproject an entire disparity map. Returns a flat list of
    (X, Y, Z) tuples with None where disparity is unusable."""
    pts = [None] * (W * H)
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            if d < min_disp:
                continue
            Z = cam.depth_from_disparity(d)
            if Z <= 0.0 or Z > max_depth:
                continue
            pts[row + x] = cam.unproject(x, y, Z)
    return pts


# =====================================================================
#  SECTION 12 : DELAUNAY TRIANGULATION (Bowyer-Watson)
#
#  Triangulating the *image-plane* coordinates and then lifting the
#  vertices to 3D is exact rather than approximate: a stereo depth map
#  is a height field over the image plane, so the 2D triangulation
#  induces a unique valid surface. No Poisson reconstruction, no
#  marching cubes, no surface-orientation ambiguity.
#
#  Delaunay specifically, because it maximises the minimum angle over
#  all triangulations of the point set. That avoids the slivers that
#  would otherwise produce degenerate, needle-shaped Gaussians in the
#  next stage.
# =====================================================================

def _circumcircle(ax, ay, bx, by, cx, cy):
    """Returns (ux, uy, r2) or None if the three points are collinear."""
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if fabs(d) < 1.0e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy)
    return (ux, uy, r2)


def delaunay(points):
    """Bowyer-Watson incremental Delaunay triangulation.

    points : list of (x, y)
    returns: list of (i, j, k) index triples into points
    """
    n = len(points)
    if n < 3:
        return []

    # --- super-triangle large enough to contain every input point ---
    minx = maxx = points[0][0]
    miny = maxy = points[0][1]
    for (x, y) in points:
        if x < minx:
            minx = x
        if x > maxx:
            maxx = x
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
    dx = maxx - minx
    dy = maxy - miny
    span = dx if dx > dy else dy
    if span <= 0.0:
        span = 1.0
    midx = 0.5 * (minx + maxx)
    midy = 0.5 * (miny + maxy)
    m = span * 20.0

    verts = list(points)
    verts.append((midx - m, midy - span))
    verts.append((midx, midy + m))
    verts.append((midx + m, midy - span))
    s0, s1, s2 = n, n + 1, n + 2

    # A triangle is stored with its circumcircle so the containment
    # test is a single distance comparison.
    def make_tri(i, j, k):
        cc = _circumcircle(verts[i][0], verts[i][1],
                           verts[j][0], verts[j][1],
                           verts[k][0], verts[k][1])
        if cc is None:
            return None
        return (i, j, k, cc[0], cc[1], cc[2])

    tris = [make_tri(s0, s1, s2)]

    for p in range(n):
        px, py = verts[p]
        bad = []
        keep = []
        for t in tris:
            ux, uy, r2 = t[3], t[4], t[5]
            ddx = px - ux
            ddy = py - uy
            if ddx * ddx + ddy * ddy < r2 - 1.0e-12:
                bad.append(t)
            else:
                keep.append(t)

        if not bad:
            continue

        # The cavity boundary is every edge used by exactly one bad
        # triangle. Counting with a dict keyed on the sorted edge.
        edge_count = {}
        for t in bad:
            i, j, k = t[0], t[1], t[2]
            for (a, b) in ((i, j), (j, k), (k, i)):
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1

        tris = keep
        for key in edge_count:
            if edge_count[key] == 1:
                nt = make_tri(key[0], key[1], p)
                if nt is not None:
                    tris.append(nt)

    # --- drop anything still touching the super-triangle ---
    out = []
    for t in tris:
        i, j, k = t[0], t[1], t[2]
        if i >= n or j >= n or k >= n:
            continue
        out.append((i, j, k))
    return out



def cull_triangles(points2d, tris, pts3d, W, max_z_ratio=1.4,
                   max_edge_factor=4.0):
    """Drop triangles that do not describe a real surface.

    Two tests, because either alone leaves artefacts.

    *Depth ratio* removes triangles straddling an occlusion boundary,
    where a foreground vertex and a background vertex would be webbed
    together into a sheet across empty space.

    *Edge length* removes the ones that survive the first test but are
    still enormous: three vertices can sit within the depth ratio of one
    another and still span half the scene if they are all far away,
    since depth resolution collapses as f*B/d flattens out. Those render
    as long spikes. The threshold is a multiple of the median edge, so
    it adapts to the scene's own scale rather than assuming metres.

    Returns the surviving index triples.
    """
    lengths = []
    prepared = []
    for (i, j, k) in tris:
        P = []
        ok = True
        for idx in (i, j, k):
            x, y = points2d[idx]
            p = pts3d[y * W + x]
            if p is None:
                ok = False
                break
            P.append(p)
        if not ok:
            continue
        zs = (P[0][2], P[1][2], P[2][2])
        zmin = fmin(zs[0], fmin(zs[1], zs[2]))
        zmax = fmax(zs[0], fmax(zs[1], zs[2]))
        if zmin <= 0.0 or zmax > zmin * max_z_ratio:
            continue
        e = 0.0
        for a in range(3):
            b = (a + 1) % 3
            dx = P[a][0] - P[b][0]
            dy = P[a][1] - P[b][1]
            dz = P[a][2] - P[b][2]
            d = fsqrt(dx * dx + dy * dy + dz * dz)
            if d > e:
                e = d
        lengths.append(e)
        prepared.append(((i, j, k), e))

    if not prepared:
        return []
    lengths.sort()
    med = lengths[len(lengths) // 2]
    limit = med * max_edge_factor
    return [t for (t, e) in prepared if e <= limit]


def fix_left_border(disp, W, H, border):
    """Repair the unmeasurable left margin of a disparity map.

    For a pixel at column x, the matching pixel in the right image sits
    at x - d. When x < d that pixel does not exist, so no disparity can
    be measured there at all, and the invalid-cost penalty that stops
    the soft-argmin picking those positions pushes the estimate toward
    zero instead. Left alone, that margin unprojects to a strip of
    geometry at the far clipping distance, curling away from the rest of
    the scene.

    Every stereo implementation has to deal with this. Here the margin
    is replaced by a horizontal extension of the first trustworthy
    column, which is what the surface behind it is almost certainly
    doing.
    """
    if border <= 0:
        return disp
    if border > W // 2:
        border = W // 2
    out = list(disp)
    for y in range(H):
        row = y * W
        anchor = disp[row + border]
        for x in range(border):
            out[row + x] = anchor
    return out


def median_filter(disp, W, H, radius=1, passes=1):
    """Median filter on the disparity map.

    Standard stereo post-processing, and it matters more here than the
    EPE suggests. A single speckled pixel barely moves the average error
    but unprojects to a spike of geometry, so the reconstruction looks
    far worse than the metric implies. The median removes isolated
    outliers while leaving depth edges sharp, which a mean filter would
    not.
    """
    cur = disp
    for _ in range(passes):
        out = [0.0] * (W * H)
        for y in range(H):
            y0 = y - radius if y - radius > 0 else 0
            y1 = y + radius if y + radius < H - 1 else H - 1
            for x in range(W):
                x0 = x - radius if x - radius > 0 else 0
                x1 = x + radius if x + radius < W - 1 else W - 1
                vals = []
                for yy in range(y0, y1 + 1):
                    row = yy * W
                    for xx in range(x0, x1 + 1):
                        vals.append(cur[row + xx])
                vals.sort()
                out[y * W + x] = vals[len(vals) // 2]
        cur = out
    return cur


def bilateral_disparity(disp, guide, W, H, radius=3, sigma_space=2.5,
                        sigma_colour=0.09, passes=2):
    """Edge-preserving smoothing of disparity, guided by the image.

    A median filter removes speckle but leaves the surface lumpy, and
    lumpiness is what a reconstruction shows most cruelly: a disparity
    error of one pixel is invisible in a heat map and obvious as a dent
    once the surface is lit and rotated.

    This averages each pixel's disparity over its neighbours, weighting
    by distance *and* by how similar the guide image is there. Within a
    surface the colours match, so it smooths hard; across a depth edge
    the colours differ, so the weight collapses and the edge survives.
    That is the same insight the refinement stage of the network uses,
    applied once more at full resolution where it is cheap.
    """
    # Precompute the spatial weights once.
    span = 2 * radius + 1
    sw = [0.0] * (span * span)
    inv_s = 1.0 / (2.0 * sigma_space * sigma_space)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sw[(dy + radius) * span + (dx + radius)] = \
                fexp(-(dx * dx + dy * dy) * inv_s)
    inv_c = 1.0 / (2.0 * sigma_colour * sigma_colour)

    # Colour weights are looked up from a small table rather than
    # calling exp() per neighbour per pixel, which would dominate.
    TABLE = 256
    ctab = [fexp(-(i / float(TABLE - 1)) ** 2 * inv_c)
            for i in range(TABLE)]

    cur = disp
    for _ in range(passes):
        out = [0.0] * (W * H)
        for y in range(H):
            y0 = -radius if y - radius >= 0 else -y
            y1 = radius if y + radius < H else H - 1 - y
            for x in range(W):
                x0 = -radius if x - radius >= 0 else -x
                x1 = radius if x + radius < W else W - 1 - x
                gc = guide[y * W + x]
                acc = 0.0
                wsum = 0.0
                for dy in range(y0, y1 + 1):
                    row = (y + dy) * W
                    srow = (dy + radius) * span + radius
                    for dx in range(x0, x1 + 1):
                        idx = row + x + dx
                        d = gc - guide[idx]
                        if d < 0.0:
                            d = -d
                        ci = int(d * (TABLE - 1))
                        if ci > TABLE - 1:
                            ci = TABLE - 1
                        w = sw[srow + dx] * ctab[ci]
                        acc += w * cur[idx]
                        wsum += w
                out[y * W + x] = acc / wsum if wsum > 0.0 else cur[y * W + x]
        cur = out
    return cur


def cull_slanted(points2d, tris, pts3d, cam, W,
                 max_slant_cos=0.07):
    """Drop triangles seen almost edge-on from the original camera.

    A depth map is a height field, so every genuine surface patch faces
    the camera to some degree. A triangle whose normal is nearly
    perpendicular to the viewing ray is not a surface: it is the
    stretched connection between a foreground vertex and the background
    behind it, and it renders as a long spike as soon as the viewpoint
    moves. This is the artefact that survives both the depth-ratio and
    edge-length culls.
    """
    out = []
    for (i, j, k) in tris:
        P = []
        ok = True
        for idx in (i, j, k):
            x, y = points2d[idx]
            p = pts3d[y * W + x]
            if p is None:
                ok = False
                break
            P.append(p)
        if not ok:
            continue
        ux = P[1][0] - P[0][0]
        uy = P[1][1] - P[0][1]
        uz = P[1][2] - P[0][2]
        vx = P[2][0] - P[0][0]
        vy = P[2][1] - P[0][1]
        vz = P[2][2] - P[0][2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        nl = fsqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1.0e-12:
            continue
        # View direction from the camera (origin) to the centroid.
        cxx = (P[0][0] + P[1][0] + P[2][0]) / 3.0
        cyy = (P[0][1] + P[1][1] + P[2][1]) / 3.0
        czz = (P[0][2] + P[1][2] + P[2][2]) / 3.0
        vl = fsqrt(cxx * cxx + cyy * cyy + czz * czz)
        if vl < 1.0e-12:
            continue
        cosang = fabs((nx * cxx + ny * cyy + nz * czz) / (nl * vl))
        if cosang < max_slant_cos:
            continue
        out.append((i, j, k))
    return out


def select_points(disp, W, H, valid, target=500, edge_fraction=0.45):
    """Choose where to place mesh vertices.

    Two populations. A uniform grid guarantees coverage of smooth
    regions, and extra samples are drawn where the depth gradient is
    large, because that is where a triangulation needs density to keep
    a depth discontinuity sharp. This adaptive placement is what later
    gives small Gaussians on detail and large flat ones on walls, with
    no explicit densification step.
    """
    # Depth gradient magnitude, forward differences.
    grad = [0.0] * (W * H)
    for y in range(H - 1):
        row = y * W
        for x in range(W - 1):
            gx = disp[row + x + 1] - disp[row + x]
            gy = disp[row + W + x] - disp[row + x]
            grad[row + x] = fsqrt(gx * gx + gy * gy)

    n_edge = int(target * edge_fraction)
    n_grid = target - n_edge

    chosen = []
    taken = [False] * (W * H)

    def add(x, y):
        i = y * W + x
        if taken[i]:
            return False
        if valid is not None and not valid[i]:
            return False
        taken[i] = True
        chosen.append((x, y))
        return True

    # Corners first: they anchor the convex hull to the image rectangle.
    for (x, y) in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        taken[y * W + x] = True
        chosen.append((x, y))

    # Uniform grid.
    step = fsqrt((W * H) / float(fmax(1, n_grid)))
    gy = 0.0
    while gy < H:
        gx = 0.0
        while gx < W:
            x = int(gx)
            y = int(gy)
            if x >= W:
                x = W - 1
            if y >= H:
                y = H - 1
            add(x, y)
            gx += step
        gy += step

    # Edge-driven samples: the strongest gradients not already taken,
    # with a minimum spacing so they do not all pile onto one contour.
    ranked = []
    for i in range(W * H):
        if grad[i] > 0.0 and not taken[i]:
            ranked.append((grad[i], i))
    ranked.sort(reverse=True)

    min_sep = 1.5
    placed = []
    for (g, i) in ranked:
        if len(placed) >= n_edge:
            break
        x = i % W
        y = i // W
        ok = True
        for (qx, qy) in placed[-40:]:
            if fabs(qx - x) < min_sep and fabs(qy - y) < min_sep:
                ok = False
                break
        if ok and add(x, y):
            placed.append((x, y))

    return chosen


# =====================================================================
#  SECTION 13 : GAUSSIAN REPRESENTATION
#
#  Each Delaunay triangle becomes one anisotropic 3D Gaussian whose
#  covariance is the *exact* covariance of a uniform density over that
#  triangle. For a triangle p0 + a*u + b*v with (a, b) uniform on the
#  unit simplex:
#
#        Var(a) = Var(b) = 1/18,     Cov(a, b) = -1/36
#
#  so, about the centroid,
#
#        S = (1/18)(u u' + v v') - (1/36)(u v' + v u')
#
#  This is rank 2 (a flat disc lying in the plane of the
#  triangle), so a small thickness is added along the surface normal to
#  make it a proper 3D Gaussian.
#
#  This is the heart of the method. Standard 3D Gaussian Splatting
#  starts from an SfM point cloud and must *discover* orientation and
#  scale through a long optimisation with heuristic splitting and
#  pruning. Here both are read directly off the geometry: a Gaussian
#  arrives already lying flat against the surface it represents, with
#  the extent of its triangle. Large flat regions produce few large
#  Gaussians, detailed regions produce many small ones, because that is
#  what the adaptive Delaunay sampling already did.
# =====================================================================

class GaussianCloud:
    """Parallel flat lists rather than objects; several times faster
    to iterate over in pure Python."""

    def __init__(self):
        self.mx = []            # mean
        self.my = []
        self.mz = []
        # base covariance, symmetric, stored as the upper triangle
        self.sxx = []
        self.sxy = []
        self.sxz = []
        self.syy = []
        self.syz = []
        self.szz = []
        self.cr = []            # colour
        self.cg = []
        self.cb = []
        self.raw_op = []        # opacity  = sigmoid(raw_op)
        self.raw_sc = []        # scale    = exp(raw_sc), multiplies S
        # gradient buffers
        self.g_cr = []
        self.g_cg = []
        self.g_cb = []
        self.g_op = []
        self.g_sc = []

    def count(self):
        return len(self.mx)

    def zero_grads(self):
        """Zero the gradient buffers *in place*.

        Rebinding these lists instead would silently break any
        optimiser holding a reference to them: the gradients would be
        written to new lists that nothing ever reads, and the loss
        would sit perfectly still.
        """
        n = self.count()
        if len(self.g_cr) != n:
            self.g_cr = [0.0] * n
            self.g_cg = [0.0] * n
            self.g_cb = [0.0] * n
            self.g_op = [0.0] * n
            self.g_sc = [0.0] * n
            return
        for i in range(n):
            self.g_cr[i] = 0.0
            self.g_cg[i] = 0.0
            self.g_cb[i] = 0.0
            self.g_op[i] = 0.0
            self.g_sc[i] = 0.0


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + fexp(-x))
    e = fexp(x)
    return e / (1.0 + e)


def gaussians_from_mesh(points2d, tris, pts3d, rgb, W, H,
                        thickness=0.35, init_opacity=0.92, scale=1.0):
    """Turn a triangulated depth map into a Gaussian cloud.

    points2d : mesh vertices in image coordinates
    tris     : index triples from delaunay()
    pts3d    : unprojected (X, Y, Z) per *pixel*
    rgb      : left colour image, flat 3*W*H
    """
    gc = GaussianCloud()
    inv_log = 0.0 if scale <= 0.0 else flog(scale)

    for (i, j, k) in tris:
        p = []
        ok = True
        for idx in (i, j, k):
            x, y = points2d[idx]
            P = pts3d[y * W + x]
            if P is None:
                ok = False
                break
            p.append(P)
        if not ok:
            continue

        # Reject triangles that straddle a depth discontinuity: they
        # would produce a huge Gaussian smeared across empty space.
        zs = [p[0][2], p[1][2], p[2][2]]
        zmin = fmin(zs[0], fmin(zs[1], zs[2]))
        zmax = fmax(zs[0], fmax(zs[1], zs[2]))
        if zmin <= 0.0 or zmax > zmin * 1.6:
            continue

        cxm = (p[0][0] + p[1][0] + p[2][0]) / 3.0
        cym = (p[0][1] + p[1][1] + p[2][1]) / 3.0
        czm = (p[0][2] + p[1][2] + p[2][2]) / 3.0

        ux = p[1][0] - p[0][0]
        uy = p[1][1] - p[0][1]
        uz = p[1][2] - p[0][2]
        vx = p[2][0] - p[0][0]
        vy = p[2][1] - p[0][1]
        vz = p[2][2] - p[0][2]

        A = 1.0 / 18.0
        B = -1.0 / 36.0
        sxx = A * (ux * ux + vx * vx) + B * 2.0 * ux * vx
        syy = A * (uy * uy + vy * vy) + B * 2.0 * uy * vy
        szz = A * (uz * uz + vz * vz) + B * 2.0 * uz * vz
        sxy = A * (ux * uy + vx * vy) + B * (ux * vy + vx * uy)
        sxz = A * (ux * uz + vx * vz) + B * (ux * vz + vx * uz)
        syz = A * (uy * uz + vy * vz) + B * (uy * vz + vy * uz)

        # Surface normal from the cross product, for the thickness term.
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        nlen = fsqrt(nx * nx + ny * ny + nz * nz)
        if nlen < 1.0e-12:
            continue
        nx /= nlen
        ny /= nlen
        nz /= nlen

        # Thickness proportional to the triangle's own in-plane size,
        # so the Gaussian stays a flattened disc rather than a sphere.
        tscale = thickness * fsqrt(fmax(sxx + syy + szz, 1.0e-12))
        t2 = tscale * tscale
        sxx += t2 * nx * nx
        syy += t2 * ny * ny
        szz += t2 * nz * nz
        sxy += t2 * nx * ny
        sxz += t2 * nx * nz
        syz += t2 * ny * nz

        # Colour: mean of the three vertex colours from the left image.
        r = g = b = 0.0
        for idx in (i, j, k):
            x, y = points2d[idx]
            o = 3 * (y * W + x)
            r += rgb[o]
            g += rgb[o + 1]
            b += rgb[o + 2]
        r /= 3.0
        g /= 3.0
        b /= 3.0

        gc.mx.append(cxm)
        gc.my.append(cym)
        gc.mz.append(czm)
        gc.sxx.append(sxx)
        gc.sxy.append(sxy)
        gc.sxz.append(sxz)
        gc.syy.append(syy)
        gc.syz.append(syz)
        gc.szz.append(szz)
        gc.cr.append(r)
        gc.cg.append(g)
        gc.cb.append(b)
        # inverse sigmoid of the initial opacity
        p0 = fclamp(init_opacity, 1.0e-4, 1.0 - 1.0e-4)
        gc.raw_op.append(flog(p0 / (1.0 - p0)))
        gc.raw_sc.append(inv_log)

    gc.zero_grads()
    return gc


# =====================================================================
#  SECTION 14 : DIFFERENTIABLE SPLAT RENDERER
#
#  EWA splatting (Zwicker et al.): project each 3D Gaussian's
#  covariance into the image with the local affine approximation of the
#  projection,
#
#        S_2D = J R S R' J'
#
#  then alpha-composite front to back. Forward and backward are both
#  written out so the cloud can be refined photometrically.
# =====================================================================

ALPHA_MAX = 0.99        # keeps the backward transmittance division stable


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """Build a world-to-camera rotation and translation."""
    fx_ = target[0] - eye[0]
    fy_ = target[1] - eye[1]
    fz_ = target[2] - eye[2]
    fl = fsqrt(fx_ * fx_ + fy_ * fy_ + fz_ * fz_)
    if fl < 1.0e-12:
        fl = 1.0
    fx_ /= fl
    fy_ /= fl
    fz_ /= fl
    # right = normalise(forward x up)
    rx = fy_ * up[2] - fz_ * up[1]
    ry = fz_ * up[0] - fx_ * up[2]
    rz = fx_ * up[1] - fy_ * up[0]
    rl = fsqrt(rx * rx + ry * ry + rz * rz)
    if rl < 1.0e-12:
        rx, ry, rz, rl = 1.0, 0.0, 0.0, 1.0
    rx /= rl
    ry /= rl
    rz /= rl
    # true up = right x forward
    ux = ry * fz_ - rz * fy_
    uy = rz * fx_ - rx * fz_
    uz = rx * fy_ - ry * fx_
    R = [rx, ry, rz, ux, uy, uz, fx_, fy_, fz_]
    t = [-(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]),
         -(R[3] * eye[0] + R[4] * eye[1] + R[5] * eye[2]),
         -(R[6] * eye[0] + R[7] * eye[1] + R[8] * eye[2])]
    return R, t


IDENTITY_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
IDENTITY_T = [0.0, 0.0, 0.0]


class SplatRenderer:
    """Renders a GaussianCloud and can push gradients back into it."""

    def __init__(self, cam, W, H, blur=0.30, cutoff=3.0):
        self.cam = cam
        self.W = W
        self.H = H
        self.blur = blur          # pixel-area regulariser on S_2D
        self.cutoff = cutoff      # Mahalanobis radius to rasterise to

    def _prepare(self, gc, R, t):
        """Project every Gaussian, returning a depth-sorted work list."""
        cam = self.cam
        W, H = self.W, self.H
        fx = cam.fx
        cx = cam.cx
        cy = cam.cy
        items = []
        n = gc.count()
        for i in range(n):
            X, Y, Z = gc.mx[i], gc.my[i], gc.mz[i]
            xc = R[0] * X + R[1] * Y + R[2] * Z + t[0]
            yc = R[3] * X + R[4] * Y + R[5] * Z + t[1]
            zc = R[6] * X + R[7] * Y + R[8] * Z + t[2]
            if zc <= 1.0e-4:
                continue

            u = fx * xc / zc + cx
            v = fx * yc / zc + cy

            # M = J R  (2x3), J the Jacobian of the pinhole projection
            f_z = fx / zc
            f_z2 = -fx / (zc * zc)
            jx = [f_z, 0.0, f_z2 * xc]
            jy = [0.0, f_z, f_z2 * yc]
            m0 = [jx[0] * R[0] + jx[1] * R[3] + jx[2] * R[6],
                  jx[0] * R[1] + jx[1] * R[4] + jx[2] * R[7],
                  jx[0] * R[2] + jx[1] * R[5] + jx[2] * R[8]]
            m1 = [jy[0] * R[0] + jy[1] * R[3] + jy[2] * R[6],
                  jy[0] * R[1] + jy[1] * R[4] + jy[2] * R[7],
                  jy[0] * R[2] + jy[1] * R[5] + jy[2] * R[8]]

            sxx, sxy, sxz = gc.sxx[i], gc.sxy[i], gc.sxz[i]
            syy, syz, szz = gc.syy[i], gc.syz[i], gc.szz[i]

            # A = M S M' is the unscaled 2D covariance
            def SM(m):
                return [sxx * m[0] + sxy * m[1] + sxz * m[2],
                        sxy * m[0] + syy * m[1] + syz * m[2],
                        sxz * m[0] + syz * m[1] + szz * m[2]]
            sm0 = SM(m0)
            sm1 = SM(m1)
            a00 = m0[0] * sm0[0] + m0[1] * sm0[1] + m0[2] * sm0[2]
            a01 = m0[0] * sm1[0] + m0[1] * sm1[1] + m0[2] * sm1[2]
            a11 = m1[0] * sm1[0] + m1[1] * sm1[1] + m1[2] * sm1[2]

            s = fexp(gc.raw_sc[i])
            c00 = s * a00 + self.blur
            c01 = s * a01
            c11 = s * a11 + self.blur

            det = c00 * c11 - c01 * c01
            if det < 1.0e-12:
                continue
            inv = 1.0 / det
            i00 = c11 * inv
            i01 = -c01 * inv
            i11 = c00 * inv

            # Largest eigenvalue of the 2x2 gives the footprint radius.
            tr = 0.5 * (c00 + c11)
            disc = fsqrt(fmax(tr * tr - det, 0.0))
            lmax = tr + disc
            rad = self.cutoff * fsqrt(fmax(lmax, 1.0e-12))
            if rad < 0.5:
                rad = 0.5
            if rad > 0.35 * (W + H):
                continue                       # pathological, drop it

            x0 = int(u - rad)
            x1 = int(u + rad) + 1
            y0 = int(v - rad)
            y1 = int(v + rad) + 1
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > W:
                x1 = W
            if y1 > H:
                y1 = H

            items.append([zc, i, u, v, i00, i01, i11,
                          x0, x1, y0, y1, a00, a01, a11, s,
                          c00, c01, c11])

        items.sort(key=lambda e: e[0])          # front to back
        return items

    def render(self, gc, R, t, background=(0.0, 0.0, 0.0)):
        W, H = self.W, self.H
        HW = W * H
        items = self._prepare(gc, R, t)

        T = [1.0] * HW
        C = [0.0] * (3 * HW)

        for e in items:
            i = e[1]
            u, v = e[2], e[3]
            i00, i01, i11 = e[4], e[5], e[6]
            x0, x1, y0, y1 = e[7], e[8], e[9], e[10]
            op = sigmoid(gc.raw_op[i])
            cr, cg, cb = gc.cr[i], gc.cg[i], gc.cb[i]
            for y in range(y0, y1):
                dy = y - v
                row = y * W
                for x in range(x0, x1):
                    Tp = T[row + x]
                    if Tp < 0.004:
                        continue
                    dx = x - u
                    q = (i00 * dx * dx + 2.0 * i01 * dx * dy +
                         i11 * dy * dy)
                    if q > 9.0:
                        continue
                    a = op * fexp(-0.5 * q)
                    if a < 1.0e-4:
                        continue
                    if a > ALPHA_MAX:
                        a = ALPHA_MAX
                    w = Tp * a
                    o = 3 * (row + x)
                    C[o] += w * cr
                    C[o + 1] += w * cg
                    C[o + 2] += w * cb
                    T[row + x] = Tp * (1.0 - a)

        br, bg, bb = background
        for p in range(HW):
            Tp = T[p]
            o = 3 * p
            C[o] += Tp * br
            C[o + 1] += Tp * bg
            C[o + 2] += Tp * bb

        self.cache = (items, T, gc, R, t)
        return C

    def backward(self, gC):
        """Back-to-front sweep accumulating gradients into the cloud.

        Maintains, per pixel, the transmittance T_i in front of Gaussian
        i (recovered by dividing out (1 - a_i), which is why alpha is
        capped) and the suffix colour S_{i+1} contributed from behind
        it. Then

            dC/dc_i = T_i * a_i
            dC/da_i = T_i * c_i - S_{i+1} / (1 - a_i)
        """
        items, Tfinal, gc, R, t = self.cache
        W, H = self.W, self.H
        HW = W * H

        T = list(Tfinal)
        S = [0.0] * (3 * HW)

        for idx in range(len(items) - 1, -1, -1):
            e = items[idx]
            i = e[1]
            u, v = e[2], e[3]
            i00, i01, i11 = e[4], e[5], e[6]
            x0, x1, y0, y1 = e[7], e[8], e[9], e[10]
            a00, a01, a11 = e[11], e[12], e[13]
            s = e[14]
            raw = gc.raw_op[i]
            op = sigmoid(raw)
            dop_draw = op * (1.0 - op)
            cr, cg, cb = gc.cr[i], gc.cg[i], gc.cb[i]

            acc_cr = 0.0
            acc_cg = 0.0
            acc_cb = 0.0
            acc_op = 0.0
            acc_sc = 0.0

            for y in range(y0, y1):
                dy = y - v
                row = y * W
                for x in range(x0, x1):
                    dx = x - u
                    q = (i00 * dx * dx + 2.0 * i01 * dx * dy +
                         i11 * dy * dy)
                    if q > 9.0:
                        continue
                    ex = fexp(-0.5 * q)
                    a = op * ex
                    if a < 1.0e-4:
                        continue
                    clipped = False
                    if a > ALPHA_MAX:
                        a = ALPHA_MAX
                        clipped = True

                    p = row + x
                    o = 3 * p
                    Ti = T[p] / (1.0 - a)
                    T[p] = Ti

                    gr, gg, gb = gC[o], gC[o + 1], gC[o + 2]

                    # colour gradient
                    w = Ti * a
                    acc_cr += gr * w
                    acc_cg += gg * w
                    acc_cb += gb * w

                    # alpha gradient
                    inv1a = 1.0 / (1.0 - a)
                    dC_da = (gr * (Ti * cr - S[o] * inv1a) +
                             gg * (Ti * cg - S[o + 1] * inv1a) +
                             gb * (Ti * cb - S[o + 2] * inv1a))

                    if not clipped:
                        acc_op += dC_da * ex * dop_draw
                        # d(alpha)/d(scale): alpha depends on scale only
                        # through q, and  dq/ds = -d' Sinv A Sinv d
                        px = i00 * dx + i01 * dy
                        py = i01 * dx + i11 * dy
                        Ap_x = a00 * px + a01 * py
                        Ap_y = a01 * px + a11 * py
                        dq_ds = -(px * Ap_x + py * Ap_y)
                        acc_sc += dC_da * a * (-0.5 * dq_ds) * s

                    # advance the suffix colour to include this Gaussian
                    S[o] += w * cr
                    S[o + 1] += w * cg
                    S[o + 2] += w * cb

            gc.g_cr[i] += acc_cr
            gc.g_cg[i] += acc_cg
            gc.g_cb[i] += acc_cb
            gc.g_op[i] += acc_op
            gc.g_sc[i] += acc_sc


# =====================================================================
#  SECTION 15 : PHOTOMETRIC REFINEMENT OF THE GAUSSIAN CLOUD
#
#  The geometry is already fixed by the Delaunay mesh, so only the
#  appearance parameters are optimised: colour, opacity and an
#  isotropic scale multiplier per Gaussian. Rendering from the original
#  left viewpoint and descending on the photometric residual closes the
#  gap between "triangles painted with vertex colours" and a
#  representation that actually reproduces the image.
# =====================================================================

class CloudAdam:
    """Adam over the Gaussian cloud's appearance parameters."""

    def __init__(self, gc, lr_colour=0.05, lr_op=0.05, lr_scale=0.02):
        self.gc = gc
        n = gc.count()
        self.groups = [(gc.cr, gc.g_cr, lr_colour),
                       (gc.cg, gc.g_cg, lr_colour),
                       (gc.cb, gc.g_cb, lr_colour),
                       (gc.raw_op, gc.g_op, lr_op),
                       (gc.raw_sc, gc.g_sc, lr_scale)]
        self.m = [[0.0] * n for _ in self.groups]
        self.v = [[0.0] * n for _ in self.groups]
        self.b1t = 1.0
        self.b2t = 1.0

    def step(self):
        b1, b2, eps = 0.9, 0.999, 1.0e-8
        self.b1t *= b1
        self.b2t *= b2
        c1 = 1.0 / (1.0 - self.b1t)
        c2 = 1.0 / (1.0 - self.b2t)
        for k in range(len(self.groups)):
            p, g, lr = self.groups[k]
            m, v = self.m[k], self.v[k]
            for i in range(len(p)):
                gi = g[i]
                if gi > 10.0:
                    gi = 10.0
                elif gi < -10.0:
                    gi = -10.0
                mi = b1 * m[i] + (1.0 - b1) * gi
                vi = b2 * v[i] + (1.0 - b2) * gi * gi
                m[i] = mi
                v[i] = vi
                p[i] -= lr * (mi * c1) / (fsqrt(vi * c2) + eps)
        # Keep colours in gamut.
        for arr in (self.gc.cr, self.gc.cg, self.gc.cb):
            for i in range(len(arr)):
                if arr[i] < 0.0:
                    arr[i] = 0.0
                elif arr[i] > 1.0:
                    arr[i] = 1.0


def refine_gaussians(gc, renderer, target_rgb, iters=60, log_every=10,
                     background=(0.0, 0.0, 0.0), verbose=True):
    """Gradient descent on the photometric residual."""
    opt = CloudAdam(gc)
    n = len(target_rgb)
    inv = 1.0 / n
    history = []
    for it in range(1, iters + 1):
        gc.zero_grads()
        C = renderer.render(gc, IDENTITY_R, IDENTITY_T, background)
        # Mean squared error and its gradient.
        loss = 0.0
        gC = [0.0] * n
        for i in range(n):
            e = C[i] - target_rgb[i]
            loss += e * e
            gC[i] = 2.0 * e * inv
        loss *= inv
        renderer.backward(gC)
        opt.step()
        if verbose and (it % log_every == 0 or it == 1):
            psnr = 10.0 * flog(1.0 / fmax(loss, 1.0e-12)) / 2.302585092994046
            print('    refine %3d   MSE %.5f   PSNR %.2f dB' %
                  (it, loss, psnr))
        history.append(loss)
    return history


# =====================================================================
#  SECTION 16 : OUTPUT: MESH FILES, NOVEL VIEWS, FULL PIPELINE
# =====================================================================

def write_obj(path, points2d, tris, pts3d, rgb, W, H, max_z_ratio=1.6):
    """Write a Wavefront OBJ with per-vertex colour.

    Vertex colours as trailing RGB on the 'v' line are understood by
    MeshLab, Blender and most viewers. Triangles spanning a depth
    discontinuity are dropped so the mesh does not web across
    occlusion boundaries.
    """
    f = open(path, 'w')
    f.write('# Reconstructed from a stereo pair\n')
    f.write('# StereoNet disparity -> Delaunay triangulation -> mesh\n')
    f.write('# %d vertices\n' % len(points2d))

    index_map = [0] * len(points2d)
    nv = 0
    for idx in range(len(points2d)):
        x, y = points2d[idx]
        P = pts3d[y * W + x]
        if P is None:
            index_map[idx] = -1
            continue
        o = 3 * (y * W + x)
        nv += 1
        index_map[idx] = nv
        # Negate Y so the mesh is right-way-up in a 3D viewer, where Y
        # points up but image rows run downward.
        f.write('v %.6f %.6f %.6f %.4f %.4f %.4f\n' %
                (P[0], -P[1], P[2], rgb[o], rgb[o + 1], rgb[o + 2]))

    nf = 0
    for (i, j, k) in tris:
        a, b, c = index_map[i], index_map[j], index_map[k]
        if a < 0 or b < 0 or c < 0:
            continue
        zs = []
        ok = True
        for idx in (i, j, k):
            x, y = points2d[idx]
            P = pts3d[y * W + x]
            if P is None:
                ok = False
                break
            zs.append(P[2])
        if not ok:
            continue
        zmin = fmin(zs[0], fmin(zs[1], zs[2]))
        zmax = fmax(zs[0], fmax(zs[1], zs[2]))
        if zmin <= 0.0 or zmax > zmin * max_z_ratio:
            continue
        f.write('f %d %d %d\n' % (a, b, c))
        nf += 1

    f.close()
    return nv, nf


def cloud_centroid(gc):
    n = gc.count()
    if n == 0:
        return (0.0, 0.0, 1.0)
    sx = sy = sz = 0.0
    for i in range(n):
        sx += gc.mx[i]
        sy += gc.my[i]
        sz += gc.mz[i]
    return (sx / n, sy / n, sz / n)


def orbit_views(gc, n_views=5, sweep=0.40, rise=0.12):
    """Camera poses on a short arc around the scene centre.

    Novel views are the proof the reconstruction is three-dimensional:
    a flat image cannot produce correct parallax, so if near objects
    slide against the background by more than far ones, the geometry
    is real.
    """
    cx, cy, cz = cloud_centroid(gc)
    radius = fsqrt(cx * cx + cy * cy + cz * cz)
    if radius < 1.0e-6:
        radius = 1.0
    views = []
    for i in range(n_views):
        t = -1.0 + 2.0 * (i / float(fmax(1, n_views - 1)))
        ang = t * sweep
        # Parametrise the eye on a circle about the centroid in the XZ
        # plane, lifted slightly so the arc is not perfectly flat.
        cos_a = fcos(ang)
        sin_a = fsqrt(fmax(0.0, 1.0 - cos_a * cos_a))
        if ang < 0.0:
            sin_a = -sin_a
        eye = (cx - radius * sin_a,
               cy + t * rise * radius,
               cz - radius * cos_a)
        views.append((ang, look_at(eye, (cx, cy, cz), (0.0, -1.0, 0.0))))
    return views


def reconstruct(disp, rgb, W, H, cam, valid=None, n_points=520,
                refine_iters=60, out_prefix='scene', n_views=5,
                verbose=True):
    """Full stage-2 pipeline: disparity in, 3D scene out."""
    if verbose:
        print('  unprojecting %d pixels' % (W * H))
    pts3d = disparity_to_points(disp, W, H, cam, min_disp=0.4)

    if verbose:
        print('  selecting mesh vertices')
    pts2d = select_points(disp, W, H, valid, target=n_points)

    if verbose:
        print('  triangulating %d vertices' % len(pts2d))
    tris = cull_triangles(pts2d, delaunay(pts2d), pts3d, W)

    nv, nf = write_obj(out_prefix + '_mesh.obj', pts2d, tris, pts3d,
                       rgb, W, H)
    if verbose:
        print('  mesh: %d vertices, %d faces -> %s_mesh.obj'
              % (nv, nf, out_prefix))

    gc = gaussians_from_mesh(pts2d, tris, pts3d, rgb, W, H)
    if verbose:
        print('  built %d anisotropic Gaussians' % gc.count())

    renderer = SplatRenderer(cam, W, H)
    if refine_iters > 0:
        if verbose:
            print('  photometric refinement:')
        refine_gaussians(gc, renderer, rgb, iters=refine_iters,
                         verbose=verbose)

    # Reconstruction from the original viewpoint.
    C0 = renderer.render(gc, IDENTITY_R, IDENTITY_T)
    write_ppm(out_prefix + '_render_original.ppm', C0, W, H)

    # Novel viewpoints the camera never occupied.
    views = orbit_views(gc, n_views=n_views)
    for vi in range(len(views)):
        ang, (R, t) = views[vi]
        Cv = renderer.render(gc, R, t)
        write_ppm('%s_novel_%d.ppm' % (out_prefix, vi), Cv, W, H)
    if verbose:
        print('  rendered %d novel views' % len(views))

    return {'gaussians': gc, 'mesh': (pts2d, tris, nv, nf),
            'render': C0, 'points3d': pts3d, 'renderer': renderer}


# =====================================================================
#  SECTION 18 : PNG DECODER
#
#  PIL is an import, so reading a real photograph means decoding PNG by
#  hand: the DEFLATE stream (RFC 1951) inside the zlib wrapper (RFC
#  1950), then the PNG per-scanline filters (RFC 2083).
#
#  PNG is lossless, so this is not an approximation. Either the bytes
#  come back exactly or the decoder is wrong, and the test at the bottom
#  of this file checks exactly that against known files.
#
#  Not supported: Adam7 interlacing (rare, and rejected with a clear
#  message rather than silently mangled).
# =====================================================================

class BitReader:
    """LSB-first bit reader over a byte string, as DEFLATE requires."""

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos
        self.buf = 0
        self.cnt = 0

    def need(self, n):
        while self.cnt < n:
            self.buf |= self.data[self.pos] << self.cnt
            self.pos += 1
            self.cnt += 8

    def bits(self, n):
        if n == 0:
            return 0
        self.need(n)
        v = self.buf & ((1 << n) - 1)
        self.buf >>= n
        self.cnt -= n
        return v

    def peek(self, n):
        self.need(n)
        return self.buf & ((1 << n) - 1)

    def drop(self, n):
        self.buf >>= n
        self.cnt -= n

    def align(self):
        drop = self.cnt & 7
        self.buf >>= drop
        self.cnt -= drop


def _reverse_bits(v, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


class Huffman:
    """Canonical Huffman decoder with a flat lookup table.

    DEFLATE packs each code most-significant-bit first into an LSB-first
    bit stream, so the bits arrive reversed relative to the code. Storing
    each symbol under the *reversed* code lets a single peek index the
    table directly, which matters: decoding bit by bit would make a
    megapixel image take minutes in interpreted Python.
    """

    def __init__(self, lengths):
        maxlen = 0
        for L in lengths:
            if L > maxlen:
                maxlen = L
        if maxlen == 0:
            self.maxlen = 0
            self.table = []
            return
        self.maxlen = maxlen

        # Canonical code assignment: count, then first code per length.
        count = [0] * (maxlen + 1)
        for L in lengths:
            if L:
                count[L] += 1
        code = 0
        next_code = [0] * (maxlen + 2)
        for L in range(1, maxlen + 1):
            code = (code + count[L - 1]) << 1
            next_code[L] = code

        size = 1 << maxlen
        table = [0] * size          # packed: symbol << 4 | length
        for sym in range(len(lengths)):
            L = lengths[sym]
            if L == 0:
                continue
            c = next_code[L]
            next_code[L] += 1
            rev = _reverse_bits(c, L)
            entry = (sym << 4) | L
            step = 1 << L
            i = rev
            while i < size:
                table[i] = entry
                i += step
        self.table = table

    def decode(self, br):
        e = self.table[br.peek(self.maxlen)]
        L = e & 15
        if L == 0:
            raise ValueError('invalid Huffman code in DEFLATE stream')
        br.drop(L)
        return e >> 4


# Length and distance tables, RFC 1951 section 3.2.5.
_LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
             35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
_LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
              3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
_DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
              257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
              8193, 12289, 16385, 24577]
_DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
               7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
_CLEN_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2,
               14, 1, 15]

_FIXED_LIT = None
_FIXED_DIST = None


def _fixed_trees():
    """The fixed Huffman trees of RFC 1951, built once and reused."""
    global _FIXED_LIT, _FIXED_DIST
    if _FIXED_LIT is None:
        lengths = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
        _FIXED_LIT = Huffman(lengths)
        _FIXED_DIST = Huffman([5] * 30)
    return _FIXED_LIT, _FIXED_DIST


def inflate(data, pos=0):
    """Decompress a raw DEFLATE stream. Returns a bytearray."""
    br = BitReader(data, pos)
    out = bytearray()

    while True:
        final = br.bits(1)
        btype = br.bits(2)

        if btype == 0:
            # Stored: byte-aligned, with a length and its complement.
            br.align()
            p = br.pos - (br.cnt >> 3)
            length = data[p] | (data[p + 1] << 8)
            nlength = data[p + 2] | (data[p + 3] << 8)
            if length != (~nlength) & 0xFFFF:
                raise ValueError('stored block length check failed')
            p += 4
            out += data[p:p + length]
            br.pos = p + length
            br.buf = 0
            br.cnt = 0

        elif btype == 1 or btype == 2:
            if btype == 1:
                lit, dist = _fixed_trees()
            else:
                hlit = br.bits(5) + 257
                hdist = br.bits(5) + 1
                hclen = br.bits(4) + 4
                clens = [0] * 19
                for i in range(hclen):
                    clens[_CLEN_ORDER[i]] = br.bits(3)
                clen_tree = Huffman(clens)

                lengths = []
                total = hlit + hdist
                while len(lengths) < total:
                    sym = clen_tree.decode(br)
                    if sym < 16:
                        lengths.append(sym)
                    elif sym == 16:
                        if not lengths:
                            raise ValueError('repeat with no previous length')
                        prev = lengths[-1]
                        for _ in range(3 + br.bits(2)):
                            lengths.append(prev)
                    elif sym == 17:
                        for _ in range(3 + br.bits(3)):
                            lengths.append(0)
                    else:
                        for _ in range(11 + br.bits(7)):
                            lengths.append(0)
                lit = Huffman(lengths[:hlit])
                dist = Huffman(lengths[hlit:total])

            # Main decode loop.
            while True:
                sym = lit.decode(br)
                if sym < 256:
                    out.append(sym)
                elif sym == 256:
                    break
                else:
                    si = sym - 257
                    length = _LEN_BASE[si] + br.bits(_LEN_EXTRA[si])
                    dsym = dist.decode(br)
                    distance = _DIST_BASE[dsym] + br.bits(_DIST_EXTRA[dsym])
                    start = len(out) - distance
                    if start < 0:
                        raise ValueError('back-reference before start of output')
                    # Overlapping copies are legal and common, so this
                    # must be a byte-at-a-time copy, not a slice.
                    if distance >= length:
                        out += out[start:start + length]
                    else:
                        for i in range(length):
                            out.append(out[start + i])
        else:
            raise ValueError('reserved DEFLATE block type 3')

        if final:
            break
    return out


def zlib_decompress(data):
    """Strip the 2-byte zlib header (RFC 1950) and inflate."""
    if len(data) < 2:
        raise ValueError('zlib stream too short')
    cmf = data[0]
    flg = data[1]
    if (cmf & 0x0F) != 8:
        raise ValueError('unsupported zlib compression method')
    if ((cmf << 8) | flg) % 31 != 0:
        raise ValueError('zlib header checksum failed')
    if flg & 0x20:
        raise ValueError('zlib preset dictionary not supported')
    return inflate(data, 2)


def _paeth(a, b, c):
    """PNG's Paeth predictor: pick whichever of left/above/upper-left is
    closest to the linear estimate a + b - c."""
    p = a + b - c
    pa = p - a
    if pa < 0:
        pa = -pa
    pb = p - b
    if pb < 0:
        pb = -pb
    pc = p - c
    if pc < 0:
        pc = -pc
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw, width, height, bpp, stride):
    """Reverse the per-scanline PNG filters. Each row is prefixed with a
    filter-type byte and may reference the row above it."""
    out = bytearray(height * stride)
    pos = 0
    prev_base = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        base = y * stride
        row = raw[pos:pos + stride]
        pos += stride

        if ftype == 0:
            out[base:base + stride] = row
        elif ftype == 1:                               # Sub
            for i in range(stride):
                a = out[base + i - bpp] if i >= bpp else 0
                out[base + i] = (row[i] + a) & 0xFF
        elif ftype == 2:                               # Up
            if y == 0:
                out[base:base + stride] = row
            else:
                for i in range(stride):
                    out[base + i] = (row[i] + out[prev_base + i]) & 0xFF
        elif ftype == 3:                               # Average
            for i in range(stride):
                a = out[base + i - bpp] if i >= bpp else 0
                b = out[prev_base + i] if y > 0 else 0
                out[base + i] = (row[i] + ((a + b) >> 1)) & 0xFF
        elif ftype == 4:                               # Paeth
            for i in range(stride):
                a = out[base + i - bpp] if i >= bpp else 0
                b = out[prev_base + i] if y > 0 else 0
                c = out[prev_base + i - bpp] if (y > 0 and i >= bpp) else 0
                out[base + i] = (row[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError('unknown PNG filter type %d on row %d'
                             % (ftype, y))
        prev_base = base
    return out


def read_png(path):
    """Decode a PNG file.

    Returns (gray, rgb, width, height) with the same contract as
    read_pnm: gray is W*H floats in [0,1], rgb is 3*W*H floats in [0,1].
    """
    f = open(path, 'rb')
    data = f.read()
    f.close()

    if data[0:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG file: ' + path)

    pos = 8
    width = height = 0
    depth = colortype = 0
    interlace = 0
    palette = None
    idat = bytearray()

    while pos < len(data):
        length = ((data[pos] << 24) | (data[pos + 1] << 16) |
                  (data[pos + 2] << 8) | data[pos + 3])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length                     # 4 len + 4 type + 4 CRC

        if ctype == b'IHDR':
            width = ((body[0] << 24) | (body[1] << 16) |
                     (body[2] << 8) | body[3])
            height = ((body[4] << 24) | (body[5] << 16) |
                      (body[6] << 8) | body[7])
            depth = body[8]
            colortype = body[9]
            interlace = body[12]
        elif ctype == b'PLTE':
            palette = body
        elif ctype == b'IDAT':
            idat += body
        elif ctype == b'IEND':
            break

    if interlace:
        raise ValueError('Adam7 interlaced PNG is not supported; '
                         're-save without interlacing')
    if depth not in (1, 2, 4, 8, 16):
        raise ValueError('unsupported PNG bit depth %d' % depth)

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colortype)
    if channels is None:
        raise ValueError('unsupported PNG colour type %d' % colortype)

    raw = _unfilter(zlib_decompress(bytes(idat)), width, height,
                    fmax(1, (channels * depth) // 8),
                    (width * channels * depth + 7) // 8)

    npix = width * height
    rgb = [0.0] * (3 * npix)
    gray = [0.0] * npix
    stride = (width * channels * depth + 7) // 8

    if depth == 16:
        inv = 1.0 / 65535.0
    else:
        inv = 1.0 / ((1 << depth) - 1)

    for y in range(height):
        base = y * stride
        for x in range(width):
            if depth == 8:
                o = base + x * channels
                s = [raw[o + c] for c in range(channels)]
            elif depth == 16:
                o = base + x * channels * 2
                s = [(raw[o + 2 * c] << 8) | raw[o + 2 * c + 1]
                     for c in range(channels)]
            else:
                # Sub-byte depths: one channel, packed MSB first.
                bit = x * depth
                byte = raw[base + (bit >> 3)]
                shift = 8 - depth - (bit & 7)
                s = [(byte >> shift) & ((1 << depth) - 1)]

            if colortype == 3:
                if palette is None:
                    raise ValueError('indexed PNG with no PLTE chunk')
                p = 3 * s[0]
                r, g, b = palette[p] / 255.0, palette[p + 1] / 255.0, \
                    palette[p + 2] / 255.0
            elif colortype in (0, 4):
                r = g = b = s[0] * inv
            else:
                r = s[0] * inv
                g = s[1] * inv
                b = s[2] * inv

            i = y * width + x
            rgb[3 * i] = r
            rgb[3 * i + 1] = g
            rgb[3 * i + 2] = b
            gray[i] = 0.299 * r + 0.587 * g + 0.114 * b

    return gray, rgb, width, height


def read_image(path):
    """Read a PNG or a Netpbm file, chosen by the magic bytes rather
    than the extension."""
    f = open(path, 'rb')
    head = f.read(8)
    f.close()
    if head[0:8] == b'\x89PNG\r\n\x1a\n':
        return read_png(path)
    return read_pnm(path)


# =====================================================================
#  SECTION 19 : PHOTO-LIKE SYNTHETIC SCENES
#
#  Random-dot stereograms prove the network is matching rather than
#  guessing, but they are texture everywhere. A real photograph has
#  blank walls, soft shading and sharp object boundaries, and a model
#  trained only on dots has never had to admit it cannot see a match.
#
#  These scenes add what dots leave out: multi-octave value noise whose
#  amplitude varies by region (so some surfaces are nearly textureless),
#  flat-shaded objects with depth-dependent brightness, and
#  disocclusions filled from the background rather than with fresh
#  noise, which is what depth-image-based rendering does and what a
#  second camera would roughly see.
# =====================================================================

def _value_noise(rng, W, H, octaves=4, persistence=0.55):
    """Multi-octave value noise: coarse random grids, bilinearly
    interpolated and summed. Gives texture at several scales at once,
    unlike the single-scale dots."""
    acc = [0.0] * (W * H)
    amp = 1.0
    total_amp = 0.0
    cells = 2
    up = BilinearUpsample()
    for _ in range(octaves):
        gw = cells + 1
        gh = cells + 1
        grid = [rng.uniform() for _ in range(gw * gh)]
        layer = up.forward(grid, 1, gh, gw, H, W)
        for i in range(W * H):
            acc[i] += amp * layer[i]
        total_amp += amp
        amp *= persistence
        cells *= 2
        if cells > W or cells > H:
            break
    inv = 1.0 / total_amp if total_amp > 0 else 1.0
    for i in range(W * H):
        acc[i] *= inv
    return acc


def _warp_to_right(left_rgb, disp, W, H, fill_from_background=True):
    """Forward-warp the left view into the right view with a z-buffer,
    then fill disocclusions.

    The z-buffer is what produces correct occlusion: where two surfaces
    project to the same right-image column, the nearer one (larger
    disparity) wins. Holes are what remains visible only to the right
    camera.
    """
    HW = W * H
    right = [-1.0] * (3 * HW)
    zbuf = [-1.0] * HW
    filled = [False] * HW
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            xr = int(x - d + 0.5)
            if xr < 0 or xr >= W:
                continue
            t = row + xr
            if d > zbuf[t]:
                zbuf[t] = d
                o = 3 * (row + x)
                right[3 * t] = left_rgb[o]
                right[3 * t + 1] = left_rgb[o + 1]
                right[3 * t + 2] = left_rgb[o + 2]
                filled[t] = True

    # Disoccluded columns: a real right camera sees background there, so
    # stretch the nearest filled pixel from the background side.
    for y in range(H):
        row = y * W
        for x in range(W):
            if filled[row + x]:
                continue
            src = -1
            for k in range(1, W):
                if x - k >= 0 and filled[row + x - k]:
                    src = row + x - k
                    break
                if x + k < W and filled[row + x + k]:
                    src = row + x + k
                    break
            if src < 0:
                continue
            o = 3 * src
            right[3 * (row + x)] = right[o]
            right[3 * (row + x) + 1] = right[o + 1]
            right[3 * (row + x) + 2] = right[o + 2]

    for i in range(3 * HW):
        if right[i] < 0.0:
            right[i] = 0.0
    return right, zbuf


def make_textured_scene(rng, W, H, max_disp=16, n_objects=3):
    """A stereo pair that looks like a photograph rather than static.

    Same return contract as make_stereogram.
    """
    HW = W * H

    # ---- depth layout ----
    disp = [0.0] * HW
    label = [0] * HW
    # Background disparity is kept close to the foreground's, because
    # depth is f*B/d and the map from disparity to depth is violently
    # non-linear at small d. A background at disparity 1 against objects
    # at 14 is not a room, it is a floor receding to the horizon: the
    # depth ratio is 14:1 and the reconstructed plane dwarfs everything
    # else. Real indoor pairs span perhaps 2:1 in depth, so the
    # generator now does too.
    back_far = rng.range(0.34, 0.46) * max_disp
    back_near = back_far * rng.range(1.15, 1.45)
    for y in range(H):
        d = back_near + (back_far - back_near) * (y / float(H - 1))
        row = y * W
        for x in range(W):
            disp[row + x] = d

    for obj in range(n_objects):
        oid = obj + 1
        ow = rng.randint(W // 5, W // 2)
        oh = rng.randint(H // 5, H // 2)
        ox = rng.randint(0, W - ow - 1)
        oy = rng.randint(0, H - oh - 1)
        d_base = rng.range(max_disp * 0.58, max_disp * 0.95)
        sx = rng.range(-2.0, 2.0) / fmax(1.0, ow)
        sy = rng.range(-2.0, 2.0) / fmax(1.0, oh)
        ellipse = rng.uniform() < 0.45
        cx = ox + ow * 0.5
        cy = oy + oh * 0.5
        rx = ow * 0.5
        ry = oh * 0.5
        for y in range(oy, oy + oh):
            row = y * W
            for x in range(ox, ox + ow):
                if ellipse:
                    nx = (x - cx) / rx
                    ny = (y - cy) / ry
                    if nx * nx + ny * ny > 1.0:
                        continue
                d = fclamp(d_base + sx * (x - ox) + sy * (y - oy),
                           0.0, float(max_disp))
                disp[row + x] = d
                label[row + x] = oid

    # ---- appearance ----
    noise = _value_noise(rng, W, H, octaves=4)
    fine = _value_noise(rng, W, H, octaves=6, persistence=0.75)

    # Each region gets its own colour and its own texture strength. A
    # low strength means a nearly textureless surface, where matching is
    # ambiguous, which is the case random dots never present.
    palette = []
    strength = []
    for i in range(n_objects + 1):
        palette.append((rng.range(0.20, 0.95),
                        rng.range(0.20, 0.95),
                        rng.range(0.20, 0.95)))
        strength.append(rng.range(0.05, 0.55))

    left_rgb = [0.0] * (3 * HW)
    for i in range(HW):
        lab = label[i]
        r, g, b = palette[lab]
        s = strength[lab]
        t = 1.0 + s * (2.0 * (0.6 * noise[i] + 0.4 * fine[i]) - 1.0)
        # Nearer surfaces are a little brighter, as under a local light.
        shade = 0.80 + 0.30 * (disp[i] / fmax(1.0, float(max_disp)))
        left_rgb[3 * i] = fclamp(r * t * shade, 0.0, 1.0)
        left_rgb[3 * i + 1] = fclamp(g * t * shade, 0.0, 1.0)
        left_rgb[3 * i + 2] = fclamp(b * t * shade, 0.0, 1.0)

    right_rgb, zbuf = _warp_to_right(left_rgb, disp, W, H)

    left = [0.0] * HW
    right = [0.0] * HW
    for i in range(HW):
        o = 3 * i
        left[i] = (0.299 * left_rgb[o] + 0.587 * left_rgb[o + 1] +
                   0.114 * left_rgb[o + 2])
        right[i] = (0.299 * right_rgb[o] + 0.587 * right_rgb[o + 1] +
                    0.114 * right_rgb[o + 2])

    valid = [False] * HW
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            xr = int(x - d + 0.5)
            if 0 <= xr < W and zbuf[row + xr] <= d + 1.0e-6:
                valid[row + x] = True

    return {'left': left, 'right': right, 'disp': disp,
            'left_rgb': left_rgb, 'right_rgb': right_rgb,
            'valid': valid, 'W': W, 'H': H, 'max_disp': max_disp}


def make_mixed_sample(rng, W, H, max_disp=16, dot_fraction=0.5):
    """Draw either a random-dot stereogram or a textured scene.

    Training on both matters. Dots force genuine binocular matching;
    textured scenes teach the network what to do where there is nothing
    to match. A model trained only on dots collapses on a blank wall.
    """
    n = rng.randint(2, 4)
    if rng.uniform() < dot_fraction:
        return make_stereogram(rng, W, H, max_disp=max_disp, n_objects=n)
    return make_textured_scene(rng, W, H, max_disp=max_disp, n_objects=n)


# =====================================================================
#  SECTION 20 : INFERENCE AT ARBITRARY RESOLUTION
#
#  The network is fully convolutional, so it runs at any size, but its
#  disparity search reaches only D*4 pixels. A real stereo pair at full
#  resolution has disparities far beyond that, so the pair is scaled
#  down until the expected disparity fits the search range, and the
#  resulting disparity map is scaled back up, values included, since
#  disparity is measured in pixels.
# =====================================================================

def resize_gray(img, W, H, W2, H2):
    up = BilinearUpsample()
    return up.forward(img, 1, H, W, H2, W2)


def resize_rgb(rgb, W, H, W2, H2):
    up = BilinearUpsample()
    planar = [0.0] * (3 * W * H)
    HW = W * H
    for i in range(HW):
        planar[i] = rgb[3 * i]
        planar[HW + i] = rgb[3 * i + 1]
        planar[2 * HW + i] = rgb[3 * i + 2]
    out = up.forward(planar, 3, H, W, H2, W2)
    HW2 = W2 * H2
    inter = [0.0] * (3 * HW2)
    for i in range(HW2):
        inter[3 * i] = out[i]
        inter[3 * i + 1] = out[HW2 + i]
        inter[3 * i + 2] = out[2 * HW2 + i]
    return inter


def _peak(disp, q=0.95):
    s = sorted(disp)
    v = s[int(len(s) * q)]
    return v if v > 0.5 else 0.5


def predict_auto(net, left, right, W, H, target_peak=None, probe_width=64,
                 max_width=320, passes=3, verbose=True):
    """Pick the working resolution from the image itself, then correct it.

    The network searches only about D*4 pixels of disparity. Run it too
    large and the true disparity leaves that window; too small and detail
    is thrown away. Neither can be fixed in advance, since the range
    depends on the scene and the baseline.

    A single low-resolution probe is not enough, and this was measured
    rather than assumed. On real Middlebury validation pairs a 64-pixel
    probe overestimated the peak disparity by 15 to 65 percent; for Baby1
    it reported 74 px against a true 45, so the pair ran at 111 pixels
    wide instead of 182 and the error more than doubled (5.12 px against
    2.26). Across the validation set, bad scale choice cost 16% of the
    error.

    So the estimate is refined: after each pass the peak is measured
    again on the prediction just made, which sits at two to three times the
    probe's resolution and is far more reliable, and the pair is rerun if
    the implied scale moved by more than 8%. It is ordinary
    coarse-to-fine, done properly.
    """
    if target_peak is None:
        target_peak = float(getattr(net, 'target_peak', 20))
    pw = probe_width if probe_width < W else W
    ph = int(H * pw / float(W) + 0.5)
    if ph < 8:
        ph = 8
    lp = resize_gray(left, W, H, pw, ph)
    rp = resize_gray(right, W, H, pw, ph)
    peak = _peak(predict(net, lp, rp, pw, ph)) * (W / float(pw))

    def width_for(pk):
        w = int(W * target_peak / fmax(pk, 1.0e-6) + 0.5)
        if w > W:
            w = W
        if w > max_width:
            w = max_width
        if w < 48:
            w = 48
        return w

    work = width_for(peak)
    if verbose:
        print('  probe at %dx%d: peak disparity ~%.1f px' % (pw, ph, peak))
    result = predict_at_scale(net, left, right, W, H, work, verbose)
    for _ in range(passes - 1):
        new_peak = _peak(result[0])
        new_work = width_for(new_peak)
        if abs(new_work - work) <= 0.08 * work:
            break
        if verbose:
            print('  re-estimated peak %.1f px -> rescaling %d -> %d'
                  % (new_peak, work, new_work))
        work = new_work
        result = predict_at_scale(net, left, right, W, H, work, verbose)
    return result


def predict_at_scale(net, left, right, W, H, work_width=96, verbose=True):
    """Run the network on a downscaled pair and return full-resolution
    disparity.

    Returns (disp_full, disp_work, W_work, H_work).
    """
    if work_width >= W:
        d = predict(net, left, right, W, H)
        return d, d, W, H

    scale = work_width / float(W)
    Ww = work_width
    Hw = int(H * scale + 0.5)
    if Hw < 8:
        Hw = 8

    lw = resize_gray(left, W, H, Ww, Hw)
    rw = resize_gray(right, W, H, Ww, Hw)
    if verbose:
        print('  running at %dx%d (input %dx%d, scale %.3f)'
              % (Ww, Hw, W, H, scale))

    dw = predict(net, lw, rw, Ww, Hw)

    # Scale back up, and scale the values with it.
    df = resize_gray(dw, Ww, Hw, W, H)
    inv = 1.0 / scale
    df = [v * inv for v in df]
    return df, dw, Ww, Hw


# =====================================================================
#  SECTION 21 : SELF-CONTAINED HTML VIEWER
#
#  Writes one .html file that opens in any browser with no server, no
#  network and no libraries: the input pair, the disparity map, and the
#  reconstructed mesh in a viewport you can drag to orbit.
#
#  The viewer is presentation only. Every number it displays was
#  computed by the pure-Python pipeline above; the JavaScript does
#  nothing but project and draw triangles that were already built. It
#  is written out by hand here, with open(), for the same reason
#  everything else is: no template library is available.
# =====================================================================

_B64 = ('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        '0123456789+/')


def b64encode(data):
    """Base64 without the base64 module. Used to embed raw pixel bytes
    in the page so the images travel inside the single file."""
    out = []
    n = len(data)
    i = 0
    while i + 2 < n:
        v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        out.append(_B64[(v >> 18) & 63])
        out.append(_B64[(v >> 12) & 63])
        out.append(_B64[(v >> 6) & 63])
        out.append(_B64[v & 63])
        i += 3
    rem = n - i
    if rem == 1:
        v = data[i] << 16
        out.append(_B64[(v >> 18) & 63])
        out.append(_B64[(v >> 12) & 63])
        out.append('==')
    elif rem == 2:
        v = (data[i] << 16) | (data[i + 1] << 8)
        out.append(_B64[(v >> 18) & 63])
        out.append(_B64[(v >> 12) & 63])
        out.append(_B64[(v >> 6) & 63])
        out.append('=')
    return ''.join(out)


def _rgb_to_b64(rgb, W, H):
    buf = bytearray()
    for i in range(W * H):
        for c in range(3):
            v = int(255.0 * rgb[3 * i + c] + 0.5)
            buf.append(0 if v < 0 else (255 if v > 255 else v))
    return b64encode(buf)


def _gray_to_b64_turbo(vals, W, H):
    """Colour-map a scalar field and return raw RGB bytes, base64'd.

    A blue-to-red ramp, because a plain greyscale depth map makes a
    1-pixel disparity error and a 10-pixel one look nearly identical.
    """
    lo = vals[0]
    hi = vals[0]
    for v in vals:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    span = hi - lo if hi > lo else 1.0
    buf = bytearray()
    for i in range(W * H):
        t = (vals[i] - lo) / span
        # piecewise blue -> cyan -> green -> yellow -> red
        if t < 0.25:
            u = t / 0.25
            r, g, b = 0.0, u, 1.0
        elif t < 0.5:
            u = (t - 0.25) / 0.25
            r, g, b = 0.0, 1.0, 1.0 - u
        elif t < 0.75:
            u = (t - 0.5) / 0.25
            r, g, b = u, 1.0, 0.0
        else:
            u = (t - 0.75) / 0.25
            r, g, b = 1.0, 1.0 - u, 0.0
        buf.append(int(255 * r))
        buf.append(int(255 * g))
        buf.append(int(255 * b))
    return b64encode(buf), lo, hi


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0e1013; --panel:#171a1f; --line:#272c34;
  --ink:#e8eaed; --dim:#9aa3af; --accent:#5aa9ff;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:16px}
h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--dim);font-size:13px;margin-bottom:16px}
.wrap{max-width:1180px;margin:0 auto}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:8px 12px;min-width:120px}
.stat b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
.stat span{color:var(--dim);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.05em}
.row{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:10px;flex:1 1 200px;min-width:180px}
.card h3{margin:0 0 8px;font-size:12px;color:var(--dim);font-weight:600;
  text-transform:uppercase;letter-spacing:.05em}
canvas{display:block;width:100%;height:auto;image-rendering:pixelated;
  border-radius:6px;background:#000}
#stage{width:100%;height:480px;image-rendering:auto;cursor:grab;
  touch-action:none}
#stage:active{cursor:grabbing}
#stagegl{width:100%;height:480px;cursor:grab;display:block;background:#000;border-radius:6px;touch-action:none}
#stagegl:active{cursor:grabbing}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  margin-top:10px}
button{background:#222831;color:var(--ink);border:1px solid var(--line);
  border-radius:7px;padding:7px 13px;font-size:13px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#06131f;border-color:var(--accent)}
label{color:var(--dim);font-size:12.5px;display:flex;align-items:center;gap:6px}
input[type=range]{accent-color:var(--accent)}
.note{color:var(--dim);font-size:12.5px;margin-top:14px;
  border-top:1px solid var(--line);padding-top:12px}
.scalebar{height:9px;border-radius:5px;margin-top:7px;
  background:linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)}
.scalelab{display:flex;justify-content:space-between;color:var(--dim);
  font-size:11px;margin-top:3px;font-variant-numeric:tabular-nums}
@media (max-width:640px){#stage,#stagegl{height:320px}}
</style>
</head>
<body><div class="wrap">
<h1>__TITLE__</h1>
<div class="sub">__SUBTITLE__</div>

<div class="stats">__STATS__</div>

<div class="row">
  <div class="card"><h3>Left input</h3><canvas id="cL"></canvas></div>
  <div class="card"><h3>Right input</h3><canvas id="cR"></canvas></div>
  <div class="card"><h3>Disparity (near = red)</h3><canvas id="cD"></canvas>
    <div class="scalebar"></div>
    <div class="scalelab"><span>__DLO__ px (far)</span><span>__DHI__ px (near)</span></div>
  </div>
</div>

<div class="card">
  <h3>Reconstruction &mdash; drag to orbit, scroll to zoom</h3>
  <canvas id="stage"></canvas>
  <canvas id="stagegl" hidden></canvas>
  <div class="controls">
    <button id="bShade" class="on">Shaded</button>
    <button id="bTex">Textured</button>
    <button id="bWire">Wireframe</button>
    <button id="bPts">Points</button>
    <button id="bSpin">Auto-rotate</button>
    <button id="bReset">Reset view</button>
    <label>FOV <input id="fov" type="range" min="20" max="90" value="55"></label>
  </div>
</div>

<div class="note">__NOTE__</div>
</div>
<script>
const W=__W__,H=__H__;
function blit(id,b64,w,h){
  const c=document.getElementById(id); c.width=w; c.height=h;
  const x=c.getContext('2d'), img=x.createImageData(w,h);
  const s=atob(b64);
  for(let i=0,j=0;i<w*h;i++){
    img.data[j++]=s.charCodeAt(i*3);
    img.data[j++]=s.charCodeAt(i*3+1);
    img.data[j++]=s.charCodeAt(i*3+2);
    img.data[j++]=255;
  }
  x.putImageData(img,0,0);
}
blit('cL',"__IMGL__",W,H); blit('cR',"__IMGR__",W,H); blit('cD',"__IMGD__",W,H);

// ---- mesh, produced by the Python pipeline ----
const V=__VERTS__;          // x,y,z triples
const C=__COLS__;           // r,g,b triples, 0-255
const F=__FACES__;          // vertex index triples
const NV=V.length/3, NF=F.length/3;

// centre and scale so the model fills the viewport
let cx=0,cy=0,cz=0;
for(let i=0;i<NV;i++){cx+=V[i*3];cy+=V[i*3+1];cz+=V[i*3+2];}
cx/=NV;cy/=NV;cz/=NV;
// Robust framing radius. A plain max() would be set by the single
// farthest outlier and shrink the real geometry to a dot.
let dists=[];
for(let i=0;i<NV;i++){
  const dx=V[i*3]-cx,dy=V[i*3+1]-cy,dz=V[i*3+2]-cz;
  dists.push(Math.sqrt(dx*dx+dy*dy+dz*dz));
}
dists.sort((a,b)=>a-b);
let rad=dists.length?dists[Math.floor(dists.length*0.92)]:1;
if(!(rad>0))rad=1;

const stage=document.getElementById('stage');
const ctx=stage.getContext('2d');
let yaw=0,pitch=-0.15,dist=rad*3.0,mode='shade',spin=false,fov=55;

function resize(){
  const r=stage.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  stage.width=Math.round(r.width*dpr); stage.height=Math.round(r.height*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
addEventListener('resize',resize);

function draw(){
  const dpr=window.devicePixelRatio||1;
  const w=stage.width/dpr, h=stage.height/dpr;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#000'; ctx.fillRect(0,0,w,h);

  const cyaw=Math.cos(yaw),syaw=Math.sin(yaw);
  const cp=Math.cos(pitch),sp=Math.sin(pitch);
  const f=(h*0.5)/Math.tan(fov*Math.PI/360);

  // rotate then translate along -z by dist
  const px=new Float32Array(NV),py=new Float32Array(NV),pz=new Float32Array(NV);
  const vx=new Float32Array(NV),vy=new Float32Array(NV),vz=new Float32Array(NV);
  for(let i=0;i<NV;i++){
    // Image rows run downward but the viewport's Y points up.
    let X=V[i*3]-cx, Y=-(V[i*3+1]-cy), Z=V[i*3+2]-cz;
    let x1= X*cyaw + Z*syaw, z1= -X*syaw + Z*cyaw;
    let y2= Y*cp - z1*sp,    z2= Y*sp + z1*cp;
    vx[i]=x1; vy[i]=y2; vz[i]=z2+dist;
    const zz=vz[i]>1e-4?vz[i]:1e-4;
    px[i]= w*0.5 + f*x1/zz;
    py[i]= h*0.5 - f*y2/zz;   // screen y grows downward
    pz[i]= zz;
  }

  if(mode==='points'){
    for(let i=0;i<NV;i++){
      if(pz[i]<=1e-4)continue;
      ctx.fillStyle='rgb('+C[i*3]+','+C[i*3+1]+','+C[i*3+2]+')';
      const s=Math.max(1,2.2*rad/pz[i]*f/200);
      ctx.fillRect(px[i]-s/2,py[i]-s/2,s,s);
    }
    return;
  }

  // painter's algorithm: farthest face first
  const order=new Array(NF), depth=new Float32Array(NF);
  for(let t=0;t<NF;t++){
    const a=F[t*3],b=F[t*3+1],c=F[t*3+2];
    depth[t]=(pz[a]+pz[b]+pz[c])/3; order[t]=t;
  }
  order.sort((p,q)=>depth[q]-depth[p]);

  for(const t of order){
    const a=F[t*3],b=F[t*3+1],c=F[t*3+2];
    if(pz[a]<=1e-4||pz[b]<=1e-4||pz[c]<=1e-4)continue;
    const ax=px[a],ay=py[a],bx=px[b],by=py[b],sx=px[c],sy=py[c];

    if(mode==='wire'){
      ctx.strokeStyle='rgba(90,169,255,0.55)'; ctx.lineWidth=0.6;
      ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
      ctx.lineTo(sx,sy);ctx.closePath();ctx.stroke();
      continue;
    }

    let r=(C[a*3]+C[b*3]+C[c*3])/3;
    let g=(C[a*3+1]+C[b*3+1]+C[c*3+1])/3;
    let bl=(C[a*3+2]+C[b*3+2]+C[c*3+2])/3;

    if(mode==='shade'){
      // Lambert against a headlight, using the view-space normal
      const ux=vx[b]-vx[a],uy=vy[b]-vy[a],uz=vz[b]-vz[a];
      const wx=vx[c]-vx[a],wy=vy[c]-vy[a],wz=vz[c]-vz[a];
      let nx=uy*wz-uz*wy, ny=uz*wx-ux*wz, nz=ux*wy-uy*wx;
      const nl=Math.hypot(nx,ny,nz)||1; nz/=nl;
      const lam=0.34+0.66*Math.abs(nz);
      r*=lam; g*=lam; bl*=lam;
    }
    ctx.fillStyle='rgb('+(r|0)+','+(g|0)+','+(bl|0)+')';
    ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
    ctx.lineTo(sx,sy);ctx.closePath();ctx.fill();
    // hairline along the shared edge kills seams between adjacent fills
    ctx.strokeStyle=ctx.fillStyle;ctx.lineWidth=0.7;ctx.stroke();
  }
}

let drag=false,lx=0,ly=0;
stage.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;
  stage.setPointerCapture(e.pointerId);});
stage.addEventListener('pointerup',e=>{drag=false;});
stage.addEventListener('pointermove',e=>{
  if(!drag)return;
  yaw+=(e.clientX-lx)*0.008; pitch+=(e.clientY-ly)*0.008;
  pitch=Math.max(-1.4,Math.min(1.4,pitch));
  lx=e.clientX;ly=e.clientY; draw();
});
stage.addEventListener('wheel',e=>{
  e.preventDefault();
  dist*=Math.exp(e.deltaY*0.0012);
  dist=Math.max(rad*0.6,Math.min(rad*14,dist)); draw();
},{passive:false});

function setMode(m,btn){
  mode=m;
  for(const id of ['bShade','bTex','bWire','bPts'])
    document.getElementById(id).classList.remove('on');
  btn.classList.add('on'); draw();
}
document.getElementById('bShade').onclick=e=>setMode('shade',e.target);
document.getElementById('bTex').onclick=e=>setMode('tex',e.target);
document.getElementById('bWire').onclick=e=>setMode('wire',e.target);
document.getElementById('bPts').onclick=e=>setMode('points',e.target);
document.getElementById('bReset').onclick=()=>{
  yaw=0;pitch=-0.15;dist=rad*3.0;draw();};
document.getElementById('fov').oninput=e=>{fov=+e.target.value;draw();};
const bs=document.getElementById('bSpin');
bs.onclick=()=>{spin=!spin;bs.classList.toggle('on',spin);if(spin)tick();};
function tick(){ if(!spin)return; yaw+=0.006; draw();
  requestAnimationFrame(tick); }
resize();
</script>
<script>
// ---- WebGL view: the photograph mapped onto a dense depth grid ----
// Opens from the left camera's own viewpoint (same focal length and
// principal point the depth was unprojected with), so the first frame is
// the photograph; dragging turns the surface around its median depth.
// Falls back to the canvas renderer above if WebGL2 is unavailable.
(function () {
  const G = __GRID__;
  if (!G) return;
  const cv = document.getElementById('stagegl');
  let gl = null; try { gl = cv.getContext('webgl2', { antialias: true }); } catch (e) {}
  if (!gl) return;
  const vs = `#version 300 es
in vec3 aPos; in vec2 aUV; in float aFill; in float aDisp;
uniform mat3 uR; uniform vec3 uPivot; uniform float uDist; uniform vec2 uS; uniform vec2 uO; uniform vec2 uNF; uniform float uPt;
out vec2 vUV; out float vFill; out float vD; out vec3 vQ;
void main() {
  vec3 q = uR * (aPos - uPivot) + vec3(0.0, 0.0, uDist);
  vQ = q; vUV = aUV; vFill = aFill; vD = aDisp;
  float n = uNF.x, f = uNF.y;
  gl_Position = vec4(uS.x * q.x + uO.x * q.z, -(uS.y * q.y + uO.y * q.z), (q.z * (f + n) - 2.0 * f * n) / (f - n), q.z);
  gl_PointSize = uPt;
}`;
  const fs = `#version 300 es
precision highp float;
in vec2 vUV; in float vFill; in float vD; in vec3 vQ;
uniform sampler2D uTex; uniform int uMode;
out vec4 o;
void main() {
  float dim = vFill > 0.5 ? 0.78 : 1.0;
  vec3 c;
  if (uMode == 1) { vec3 nrm = normalize(cross(dFdx(vQ), dFdy(vQ))); c = vec3(0.80, 0.82, 0.86) * (0.22 + 0.78 * abs(dot(nrm, normalize(-vQ)))); }
  else if (uMode == 2) { c = vec3(0.24, 0.71, 0.77); dim = 1.0; }
  else c = texture(uTex, vUV).rgb;
  o = vec4(c * dim, 1.0);
}`;
  const sh = (t, src) => { const x = gl.createShader(t); gl.shaderSource(x, src); gl.compileShader(x);
    if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  let prog;
  try { prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog)); }
  catch (e) { console.warn('WebGL unavailable: ' + e.message); return; }
  document.getElementById('stage').hidden = true; cv.hidden = false;
  const fovEl = document.getElementById('fov'); if (fovEl) fovEl.parentElement.hidden = true;
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const attr = (name, arr, size) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, name); if (loc >= 0) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); } };
  attr('aPos', G.pos, 3); attr('aUV', G.uv, 2); attr('aFill', G.fill, 1); attr('aDisp', G.disp, 1);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(G.idx), gl.STATIC_DRAW);
  // wireframe: triangle edges of every 4th cell row would be dense; use all edges of the index list, subsampled
  const L = []; for (let t = 0; t < G.idx.length; t += 3 * 8) { const a = G.idx[t], b = G.idx[t + 1], c = G.idx[t + 2]; L.push(a, b, b, c, c, a); }
  const lb = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lb); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(L), gl.STATIC_DRAW);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, document.getElementById('cL'));
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  const U = n => gl.getUniformLocation(prog, n);
  const st = { yaw: 0, pitch: 0, dist: G.pivot, mode: 1, spin: false };
  function render() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const Wc = Math.max(1, Math.round(r.width * dpr)), Hc = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== Wc || cv.height !== Hc) { cv.width = Wc; cv.height = Hc; }
    gl.viewport(0, 0, Wc, Hc); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const s = Math.min(Wc / G.W, Hc / G.H), F = G.fx * s;
    const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.enable(gl.DEPTH_TEST);
    gl.uniformMatrix3fv(U('uR'), false, [cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy]);
    gl.uniform3f(U('uPivot'), 0, 0, G.pivot); gl.uniform1f(U('uDist'), st.dist);
    gl.uniform2f(U('uS'), 2 * F / Wc, 2 * F / Hc);
    gl.uniform2f(U('uO'), 2 * (G.cx - G.W / 2) * s / Wc, 2 * (G.cy - G.H / 2) * s / Hc);
    gl.uniform2f(U('uNF'), G.pivot * 0.01, G.pivot * 100); gl.uniform1f(U('uPt'), 2 * dpr);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(U('uTex'), 0);
    gl.uniform1i(U('uMode'), st.mode === 3 ? 0 : st.mode);
    if (st.mode === 2) { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lb); gl.drawElements(gl.LINES, L.length, gl.UNSIGNED_INT, 0); }
    else if (st.mode === 3) gl.drawArrays(gl.POINTS, 0, G.pos.length / 3);
    else { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.drawElements(gl.TRIANGLES, G.idx.length, gl.UNSIGNED_INT, 0); }
  }
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointerup', () => { drag = null; });
  cv.addEventListener('pointermove', e => { if (!drag) return;
    st.yaw += (e.clientX - drag[0]) * 0.005; st.pitch = Math.max(-1.3, Math.min(1.3, st.pitch + (e.clientY - drag[1]) * 0.005));
    drag = [e.clientX, e.clientY]; render(); });
  cv.addEventListener('wheel', e => { e.preventDefault(); st.dist = Math.max(G.pivot * 0.25, Math.min(G.pivot * 4, st.dist * Math.exp(e.deltaY * 0.0012))); render(); }, { passive: false });
  const on = (id, f) => { const b = document.getElementById(id); if (b) b.addEventListener('click', f); };
  on('bShade', () => { st.mode = 1; render(); }); on('bTex', () => { st.mode = 0; render(); });
  on('bWire', () => { st.mode = 2; render(); }); on('bPts', () => { st.mode = 3; render(); });
  on('bReset', () => { st.yaw = 0; st.pitch = 0; st.dist = G.pivot; render(); });
  on('bSpin', () => { st.spin = !st.spin; const loop = () => { if (!st.spin) return; st.yaw = Math.sin(performance.now() / 2400) * 0.35; render(); requestAnimationFrame(loop); }; loop(); });
  // open on the photograph itself
  st.mode = 0;
  for (const id of ['bShade', 'bTex', 'bWire', 'bPts']) { const b = document.getElementById(id); if (b) b.classList.toggle('on', id === 'bTex'); }
  const rb = document.getElementById('bReset'); if (rb) rb.textContent = 'Camera view';
  new ResizeObserver(render).observe(cv); render();
})();
</script>
</body></html>
"""


def write_html_viewer(path, left_rgb, right_rgb, disp, W, H,
                      points2d, tris, pts3d, rgb_for_mesh,
                      title='Stereo 3D reconstruction',
                      subtitle='', stats=None, note='',
                      max_z_ratio=1.6, mesh_W=None, grid=None):
    """Emit the single-file viewer. mesh_W is the width of the canvas the
    mesh vertices index into (2W when a background layer is merged)."""
    Wm = W if mesh_W is None else mesh_W
    imgL = _rgb_to_b64(left_rgb, W, H)
    imgR = _rgb_to_b64(right_rgb, W, H)
    imgD, dlo, dhi = _gray_to_b64_turbo(disp, W, H)

    # Build the mesh arrays, dropping triangles that span a depth jump.
    index_map = [-1] * len(points2d)
    verts = []
    cols = []
    for idx in range(len(points2d)):
        x, y = points2d[idx]
        P = pts3d[y * Wm + x]
        if P is None:
            continue
        index_map[idx] = len(verts) // 3
        verts.append(P[0])
        verts.append(P[1])
        verts.append(P[2])
        o = 3 * (y * Wm + x)
        cols.append(int(255 * fclamp(rgb_for_mesh[o], 0.0, 1.0)))
        cols.append(int(255 * fclamp(rgb_for_mesh[o + 1], 0.0, 1.0)))
        cols.append(int(255 * fclamp(rgb_for_mesh[o + 2], 0.0, 1.0)))

    faces = []
    for (i, j, k) in tris:
        a, b, c = index_map[i], index_map[j], index_map[k]
        if a < 0 or b < 0 or c < 0:
            continue
        zs = []
        ok = True
        for idx in (i, j, k):
            x, y = points2d[idx]
            P = pts3d[y * Wm + x]
            if P is None:
                ok = False
                break
            zs.append(P[2])
        if not ok:
            continue
        zmin = fmin(zs[0], fmin(zs[1], zs[2]))
        zmax = fmax(zs[0], fmax(zs[1], zs[2]))
        if zmin <= 0.0 or zmax > zmin * max_z_ratio:
            continue
        faces.append(a)
        faces.append(b)
        faces.append(c)

    def jsnum(v):
        return '%.5g' % v

    stat_html = ''
    if stats:
        for (lab, val) in stats:
            stat_html += ('<div class="stat"><b>%s</b><span>%s</span></div>'
                          % (val, lab))

    html = _HTML_TEMPLATE
    html = html.replace('__TITLE__', title)
    html = html.replace('__SUBTITLE__', subtitle)
    html = html.replace('__STATS__', stat_html)
    html = html.replace('__NOTE__', note)
    html = html.replace('__W__', str(W))
    html = html.replace('__H__', str(H))
    html = html.replace('__DLO__', '%.1f' % dlo)
    html = html.replace('__DHI__', '%.1f' % dhi)
    html = html.replace('__IMGL__', imgL)
    html = html.replace('__IMGR__', imgR)
    html = html.replace('__IMGD__', imgD)
    html = html.replace('__VERTS__',
                        '[' + ','.join([jsnum(v) for v in verts]) + ']')
    html = html.replace('__COLS__',
                        '[' + ','.join([str(v) for v in cols]) + ']')
    html = html.replace('__FACES__',
                        '[' + ','.join([str(v) for v in faces]) + ']')
    if grid is None:
        html = html.replace('__GRID__', 'null')
    else:
        html = html.replace('__GRID__', (
            '{"W":%d,"H":%d,"fx":%s,"cx":%s,"cy":%s,"pivot":%s,'
            '"pos":[%s],"uv":[%s],"fill":[%s],"disp":[%s],"idx":[%s]}'
            % (grid['W'], grid['H'], jsnum(grid['fx']), jsnum(grid['cx']),
               jsnum(grid['cy']), jsnum(grid['pivot']),
               ','.join([jsnum(v) for v in grid['pos']]),
               ','.join(['%.4f' % v for v in grid['uv']]),
               ','.join([str(v) for v in grid['fill']]),
               ','.join(['%.2f' % v for v in grid['disp']]),
               ','.join([str(v) for v in grid['idx']]))))

    f = open(path, 'w')
    f.write(html)
    f.close()
    return len(verts) // 3, len(faces) // 3


# =====================================================================
#  SECTION 22 : FILE-DRIVEN DEMO
#
#  Point it at two image files and it produces the whole reconstruction
#  plus a viewer page. Three resolutions are in play and keeping them
#  straight matters:
#
#    native   - whatever the files are
#    work     - what the network runs at, small enough that the real
#               disparity falls inside the D*4 pixel search range
#    display  - what the mesh, Gaussians and viewer use
#
#  Disparity is in pixels, so rescaling the map between resolutions
#  rescales its values too, and the focal length must be rescaled with
#  the image or the reconstruction comes out the wrong shape.
# =====================================================================

def reconstruct_from_pair(net, left_gray, left_rgb, right_gray, right_rgb,
                          W, H, out_prefix, cam=None, calib=None,
                          work_width=None, display_width=480,
                          refine_iters=90, n_points=None, n_views=5,
                          consistency=True, refine=True,
                          fill_background=True, refine_max_width=640,
                          gt_disp=None, gt_valid=None, title=None,
                          verbose=True):
    """Full pipeline from an in-memory stereo pair.

    refine          : full-resolution band SGM + planes (section 25)
    fill_background : extra layer behind foreground edges, so rotated
                      views do not tear (completion, not measurement)
    """
    if verbose:
        print('input %dx%d' % (W, H))

    # ---- stage 1: disparity ----
    if refine:
        disp_full, disp_work, Ww, Hw = predict_auto(
            net, left_gray, right_gray, W, H, verbose=verbose)
        if W <= refine_max_width:
            disp_full, _status = refine_disparity(
                left_gray, right_gray, left_rgb, disp_full, W, H,
                verbose=verbose)
        else:
            # Large photographs: refine at refine_max_width, which keeps
            # pure Python to seconds, then bring the result back up.
            Wr = refine_max_width
            sr = Wr / float(W)
            Hr = int(H * sr + 0.5)
            dr, _status = refine_disparity(
                resize_gray(left_gray, W, H, Wr, Hr),
                resize_gray(right_gray, W, H, Wr, Hr),
                resize_rgb(left_rgb, W, H, Wr, Hr),
                [v * sr for v in resize_gray(disp_full, W, H, Wr, Hr)],
                Wr, Hr, verbose=verbose)
            disp_full = [v / sr for v in resize_gray(dr, Wr, Hr, W, H)]
            if verbose:
                print('  (refined at %dx%d)' % (Wr, Hr))
    elif consistency:
        disp_full, _frac = predict_consistent(
            net, left_gray, right_gray, W, H, work_width, verbose=verbose)
        Ww, Hw = W, H
    elif work_width is None:
        disp_full, disp_work, Ww, Hw = predict_auto(
            net, left_gray, right_gray, W, H, verbose=verbose)
    else:
        disp_full, disp_work, Ww, Hw = predict_at_scale(
            net, left_gray, right_gray, W, H, work_width, verbose)

    # ---- choose the display resolution ----
    Wd = display_width if display_width < W else W
    scale_d = Wd / float(W)
    Hd = int(H * scale_d + 0.5)
    if Wd == W:
        dispd = disp_full
        rgbd = left_rgb
        rgbrd = right_rgb
    else:
        dispd = resize_gray(disp_full, W, H, Wd, Hd)
        dispd = [v * scale_d for v in dispd]     # disparity is in pixels
        rgbd = resize_rgb(left_rgb, W, H, Wd, Hd)
        rgbrd = resize_rgb(right_rgb, W, H, Wd, Hd)
    if verbose:
        print('  display %dx%d   disparity %.2f..%.2f px'
              % (Wd, Hd, min(dispd), max(dispd)))

    # ---- camera, rescaled to the display resolution ----
    if cam is None:
        if calib is not None:
            c = read_calib(calib)
            fx = c['fx'] * scale_d
            cam = Camera(fx, c['cx'] * scale_d, c['cy'] * scale_d,
                         c['baseline'], c['doffs'] * scale_d)
            if verbose:
                print('  calibration from file: fx %.1f  baseline %.3f'
                      % (cam.fx, cam.baseline))
        else:
            cam = Camera.synthetic(Wd, Hd, baseline=0.10)
            if verbose:
                print('  no calibration supplied, assuming a 60 degree '
                      'field of view and a 10 cm baseline')
                print('  (geometry is correct up to an overall scale)')

    metrics_txt = None
    if gt_disp is not None:
        gt_d = gt_disp
        gt_v = gt_valid if gt_valid is not None else [True] * (W * H)
        epe, b1, b3 = metrics(disp_full, gt_d, gt_v)
        metrics_txt = (epe, b1, b3)
        if verbose:
            print('  EPE %.3f px   bad1.0 %.1f%%   bad3.0 %.1f%%'
                  % (epe, b1, b3))

    write_pgm(out_prefix + '_disparity.pgm', dispd, Wd, Hd)

    # ---- stage 2: geometry, mesh, Gaussians ----
    # The left margin is unmeasurable by construction; repair it before
    # any filtering, so its bogus values are not smeared inward.
    # (Refinement already handles it: pixels whose match would fall off
    # the right image are filled from their farther neighbour.)
    if not refine:
        border = int(max(dispd) + 2.0)
        dispd = fix_left_border(dispd, Wd, Hd, border)
        if verbose:
            print('  repaired %d-pixel unmeasurable left margin' % border)

    # Smoothing helps the raw network output but blurs a refined map:
    # measured on validation, it doubled the Chamfer distance of the
    # refined mesh (1.52 -> 2.65), so it only runs without refinement.
    if not refine:
        dispd = median_filter(dispd, Wd, Hd, radius=1, passes=2)
        gd = [0.0] * (Wd * Hd)
        for i in range(Wd * Hd):
            gd[i] = (0.299 * rgbd[3 * i] + 0.587 * rgbd[3 * i + 1] +
                     0.114 * rgbd[3 * i + 2])
        dispd = bilateral_disparity(dispd, gd, Wd, Hd, radius=4, passes=3)
        if verbose:
            print('  median + image-guided bilateral filtering applied')

    # Mesh density scales with the image. A fixed vertex budget gave a
    # 320x240 reconstruction the same 650 vertices as a 96x72 one, which
    # is far too coarse: the triangles end up larger than the objects,
    # so every surface reads as faceted no matter how good the disparity
    # underneath it is.
    if n_points is None:
        n_points = int((Wd * Hd) / 40)
        if n_points < 600:
            n_points = 600
        if n_points > 5000:
            n_points = 5000
        if verbose:
            print('  mesh budget %d vertices for %dx%d' % (n_points, Wd, Hd))

    dmin = robust_min_disparity(dispd)
    if verbose:
        print('  depth clamp: disparity floor %.2f px (max depth %.2f m)'
              % (dmin, cam.depth_from_disparity(dmin)))
    pts3d = disparity_to_points(dispd, Wd, Hd, cam, min_disp=dmin)
    pts2d = select_points(dispd, Wd, Hd, None, target=n_points)
    tris = delaunay(pts2d)
    n_raw = len(tris)
    tris = cull_triangles(pts2d, tris, pts3d, Wd)
    n_mid = len(tris)
    tris = cull_slanted(pts2d, tris, pts3d, cam, Wd)
    if verbose:
        print('  triangulation %d -> %d (depth/size) -> %d (edge-on)'
              % (n_raw, n_mid, len(tris)))
    Wm = Wd
    mrgb = rgbd
    if fill_background:
        layer = background_layer(dispd, rgbd, Wd, Hd, cam, n_points,
                                 min_disp=dmin)
        if verbose:
            print('  background layer: %d triangles behind foreground edges'
                  % len(layer[1]))
        pts2d, tris, pts3d, mrgb, Wm = merge_layer(pts2d, tris, pts3d, rgbd,
                                                   layer, Wd, Hd)
    nv, nf = write_obj(out_prefix + '_mesh.obj', pts2d, tris, pts3d,
                       mrgb, Wm, Hd)
    if verbose:
        print('  mesh %d vertices, %d faces' % (nv, nf))

    gc = gaussians_from_mesh(pts2d, tris, pts3d, mrgb, Wm, Hd)
    renderer = SplatRenderer(cam, Wd, Hd)
    psnr = None
    if refine_iters > 0 and gc.count() > 0:
        hist = refine_gaussians(gc, renderer, rgbd, iters=refine_iters,
                                log_every=max(1, refine_iters // 3),
                                verbose=verbose)
        psnr = 10.0 * flog(1.0 / fmax(hist[-1], 1.0e-12)) / 2.302585092994046
    if verbose:
        print('  %d Gaussians' % gc.count())

    C0 = renderer.render(gc, IDENTITY_R, IDENTITY_T)
    write_ppm(out_prefix + '_render_original.ppm', C0, Wd, Hd)
    for vi, (ang, (R, t)) in enumerate(orbit_views(gc, n_views=n_views)):
        write_ppm('%s_novel_%d.ppm' % (out_prefix, vi),
                  renderer.render(gc, R, t), Wd, Hd)

    # ---- viewer ----
    stats = [('Input', '%dx%d' % (W, H)),
             ('Mesh', '%d tri' % nf),
             ('Gaussians', '%d' % gc.count())]
    if psnr is not None:
        stats.append(('Gaussian fit', '%.1f dB' % psnr))
    if metrics_txt is not None:
        stats.append(('EPE', '%.2f px' % metrics_txt[0]))
        stats.append(('bad 3.0', '%.1f%%' % metrics_txt[2]))

    note = ('Built with no imports: the disparity network, its '
            'backpropagation, the PNG decoder, the Delaunay '
            'triangulation and the Gaussian splatting are all plain '
            'Python. This page is presentation only &mdash; every number '
            'above and every triangle below was computed by that code, '
            'and the JavaScript here does nothing but project and draw '
            'them.')

    html_path = out_prefix + '_viewer.html'
    grid = grid_mesh(dispd, Wd, Hd, cam, fill=fill_background,
                     min_disp=dmin)
    if verbose:
        print('  viewer grid: %d vertices, %d triangles textured with the '
              'photograph (+%d filled in)' % (len(grid['disp']),
                                              grid['main_tris'],
                                              grid['fill_tris']))
    write_html_viewer(html_path, rgbd, rgbrd, dispd, Wd, Hd,
                      pts2d, tris, pts3d, mrgb, mesh_W=Wm, grid=grid,
                      title=title or 'Stereo 3D reconstruction',
                      subtitle=('%dx%d input, network run at %dx%d, '
                                '%d triangles, %d Gaussians'
                                % (W, H, Ww, Hw, nf, gc.count())),
                      stats=stats, note=note)
    if verbose:
        print('  viewer -> %s' % html_path)

    return {'disp': dispd, 'W': Wd, 'H': Hd, 'cam': cam, 'gaussians': gc,
            'mesh': (pts2d, tris, nv, nf), 'html': html_path,
            'metrics': metrics_txt, 'psnr': psnr}


def demo_files(left_path, right_path, out_prefix='demo',
               weights='stereonet_weights.txt', calib=None,
               work_width=None, display_width=480, refine_iters=90,
               consistency=True):
    """Reconstruct from two image files on disk (PNG or PGM/PPM)."""
    net = load_model(weights)

    lg, lrgb, W, H = read_image(left_path)
    rg, rrgb, W2, H2 = read_image(right_path)
    if W != W2 or H != H2:
        raise ValueError('the two images must be the same size (%dx%d vs '
                         '%dx%d)' % (W, H, W2, H2))

    return reconstruct_from_pair(net, lg, lrgb, rg, rrgb, W, H,
                                 out_prefix, calib=calib,
                                 work_width=work_width,
                                 display_width=display_width,
                                 refine_iters=refine_iters,
                                 consistency=consistency,
                                 title='Stereo 3D reconstruction')


def demo_synthetic(out_prefix='demo', weights='stereonet_weights.txt',
                   seed=7, W=160, H=120, max_disp=14, refine_iters=90):
    """Reconstruct a generated scene. Works with no input files, so the
    demo cannot fail for want of a dataset."""
    net = load_model(weights)
    s = make_textured_scene(Rng(seed), W, H, max_disp=max_disp,
                            n_objects=3)
    return reconstruct_from_pair(net, s['left'], s['left_rgb'],
                                 s['right'], s['right_rgb'], W, H,
                                 out_prefix, work_width=None,
                                 display_width=W,
                                 refine_iters=refine_iters,
                                 gt_disp=s['disp'], gt_valid=s['valid'],
                                 title='Stereo 3D reconstruction '
                                       '(generated scene)')


# =====================================================================
#  SECTION 23 : RENDERED-SCENE DATASET
#
#  Loads physically ray-traced stereo pairs from disk. Both views are
#  rendered independently from 3D geometry, so unlike the warp-based
#  generators these carry real perspective, real occlusion and real
#  shading -- the things a 2D warp cannot fake.
#
#  The folder carries an index.txt listing its scenes, because listing a
#  directory would need os.listdir, and that is an import.
# =====================================================================

def read_scene_index(dirpath):
    """Return the list of scene prefixes in a rendered dataset folder."""
    if not dirpath.endswith('/'):
        dirpath = dirpath + '/'
    f = open(dirpath + 'index.txt', 'r')
    lines = f.read().split('\n')
    f.close()
    n = int(lines[0])
    return dirpath, [lines[i + 1].strip() for i in range(n)
                     if lines[i + 1].strip()]


def occlusion_validity(disp, W, H):
    """Which left pixels are actually visible in the right image.

    A left pixel at x maps to x-d on the right. Where two surfaces land
    on the same right column the nearer one (larger disparity) wins and
    the other is occluded -- no stereo method can recover it, so it must
    be excluded from both the loss and the metrics. Scoring occluded
    pixels rewards whatever a model happens to hallucinate there.
    """
    HW = W * H
    zbuf = [-1.0] * HW
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            xr = int(x - d + 0.5)
            if 0 <= xr < W and d > zbuf[row + xr]:
                zbuf[row + xr] = d
    valid = [False] * HW
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp[row + x]
            xr = int(x - d + 0.5)
            if 0 <= xr < W and zbuf[row + xr] <= d + 1.0e-6:
                valid[row + x] = True
    return valid


def load_rendered(dirpath, prefix, disp_scale=64.0):
    """One rendered scene, in the same shape make_stereogram returns."""
    if not dirpath.endswith('/'):
        dirpath = dirpath + '/'
    lg, lrgb, W, H = read_image(dirpath + prefix + '_left.png')
    rg, rrgb, W2, H2 = read_image(dirpath + prefix + '_right.png')
    dg, _, W3, H3 = read_pnm(dirpath + prefix + '_disp.pgm')
    if (W, H) != (W2, H2) or (W, H) != (W3, H3):
        raise ValueError('scene %s has mismatched sizes' % prefix)

    # The PGM stores disparity * scale in 16 bits; 0 means unknown.
    disp = [0.0] * (W * H)
    known = [False] * (W * H)
    for i in range(W * H):
        raw = dg[i] * 65535.0
        if raw < 0.5:
            continue
        disp[i] = raw / disp_scale
        known[i] = True

    valid = occlusion_validity(disp, W, H)
    for i in range(W * H):
        if not known[i]:
            valid[i] = False

    return {'left': lg, 'right': rg, 'disp': disp, 'left_rgb': lrgb,
            'right_rgb': rrgb, 'valid': valid, 'W': W, 'H': H,
            'max_disp': 16}


def crop_sample(sample, cw, ch, rng):
    """Random crop, so one rendered scene yields many training examples.

    The crop must not start at x=0 more often than chance, or the network
    over-learns the unmeasurable left margin.
    """
    W, H = sample['W'], sample['H']
    if cw >= W and ch >= H:
        return sample
    x0 = rng.randint(0, W - cw) if W > cw else 0
    y0 = rng.randint(0, H - ch) if H > ch else 0
    out = {'W': cw, 'H': ch, 'max_disp': sample['max_disp']}
    for key in ('left', 'right', 'disp'):
        src = sample[key]
        dst = [0.0] * (cw * ch)
        for y in range(ch):
            srow = (y0 + y) * W + x0
            drow = y * cw
            for x in range(cw):
                dst[drow + x] = src[srow + x]
        out[key] = dst
    v = sample['valid']
    dv = [False] * (cw * ch)
    for y in range(ch):
        srow = (y0 + y) * W + x0
        drow = y * cw
        for x in range(cw):
            dv[drow + x] = v[srow + x]
    out['valid'] = dv
    for key in ('left_rgb', 'right_rgb'):
        if sample.get(key) is None:
            out[key] = None
            continue
        src = sample[key]
        dst = [0.0] * (3 * cw * ch)
        for y in range(ch):
            srow = (y0 + y) * W + x0
            drow = y * cw
            for x in range(cw):
                for c in range(3):
                    dst[3 * (drow + x) + c] = src[3 * (srow + x) + c]
        out[key] = dst
    return out


class RenderedSampler:
    """Draws crops from a folder of rendered scenes, mixed with the
    procedural generators.

    The mix matters. Random dots force genuine binocular matching with no
    monocular cue to fall back on; the rendered scenes supply real
    perspective, occlusion and blank surfaces. Trained on either alone
    the model overfits to that world's statistics.
    """

    def __init__(self, dirpath, dot_fraction=0.25, warp_fraction=0.15,
                 cache=24):
        self.dir, self.names = read_scene_index(dirpath)
        self.dot_fraction = dot_fraction
        self.warp_fraction = warp_fraction
        self.cache = {}
        self.cache_limit = cache
        self.order = []

    def _scene(self, rng):
        name = self.names[rng.randint(0, len(self.names) - 1)]
        if name in self.cache:
            return self.cache[name]
        s = load_rendered(self.dir, name)
        self.cache[name] = s
        self.order.append(name)
        if len(self.order) > self.cache_limit:
            drop = self.order.pop(0)
            if drop in self.cache:
                del self.cache[drop]
        return s

    def __call__(self, rng, W, H, max_disp):
        u = rng.uniform()
        if u < self.dot_fraction:
            return make_stereogram(rng, W, H, max_disp=max_disp,
                                   n_objects=rng.randint(2, 4))
        if u < self.dot_fraction + self.warp_fraction:
            return make_textured_scene(rng, W, H, max_disp=max_disp,
                                       n_objects=rng.randint(2, 4))
        return crop_sample(self._scene(rng), W, H, rng)


# =====================================================================
#  SECTION 24 : LEFT-RIGHT CONSISTENCY
#
#  A disparity map computed from the left image alone cannot tell a good
#  match from a confident guess. Where a surface is occluded in the right
#  view there is no correct answer, and the network will still produce
#  one -- usually the background stretched across the gap, which is
#  precisely the artefact that survives every geometric cull downstream.
#
#  The classical remedy is to compute disparity twice, once with each
#  image as reference, and keep only the pixels where the two agree:
#
#      | d_L(x) - d_R(x - d_L(x)) |  <=  tolerance
#
#  Disagreement means occlusion or a mismatch. Those pixels are dropped
#  and refilled from the nearest agreeing neighbour, taking the smaller
#  disparity of the two sides, because a disoccluded region belongs to
#  the background rather than the object that hid it.
#
#  The network is trained with the left image as reference, so the right
#  map is obtained by mirroring both images, running it, and mirroring
#  the result back. Mirroring turns a right-reference problem into a
#  left-reference one exactly.
# =====================================================================

def mirror_h(img, C, W, H):
    """Flip an image horizontally, channel-planar layout."""
    out = [0.0] * (C * W * H)
    for c in range(C):
        cb = c * W * H
        for y in range(H):
            row = cb + y * W
            for x in range(W):
                out[row + x] = img[row + (W - 1 - x)]
    return out


def lr_consistency(disp_l, disp_r, W, H, tol=1.5):
    """Mark pixels where the two disparity maps disagree.

    disp_r must already be expressed in the right image's own frame.
    """
    valid = [True] * (W * H)
    for y in range(H):
        row = y * W
        for x in range(W):
            d = disp_l[row + x]
            xr = x - d
            if xr < 0.0 or xr > W - 1:
                valid[row + x] = False
                continue
            x0 = int(xr)
            x1 = x0 + 1 if x0 + 1 < W else x0
            f = xr - x0
            dr = disp_r[row + x0] * (1.0 - f) + disp_r[row + x1] * f
            diff = d - dr
            if diff < 0.0:
                diff = -diff
            if diff > tol:
                valid[row + x] = False
    return valid


def fill_invalid(disp, valid, W, H):
    """Refill rejected pixels from the nearest agreeing neighbour on the
    same row, choosing the smaller disparity of the two sides.

    A region disoccluded in the right view is background revealed by the
    foreground object's edge, so the background's disparity is the right
    answer -- taking the larger one would smear the object outward, which
    is the classic foreground-fattening artefact.
    """
    out = list(disp)
    for y in range(H):
        row = y * W
        x = 0
        while x < W:
            if valid[row + x]:
                x += 1
                continue
            start = x
            while x < W and not valid[row + x]:
                x += 1
            end = x                      # first valid at or after start
            left_val = disp[row + start - 1] if start > 0 else None
            right_val = disp[row + end] if end < W else None
            if left_val is None and right_val is None:
                continue
            if left_val is None:
                fillv = right_val
            elif right_val is None:
                fillv = left_val
            else:
                fillv = left_val if left_val < right_val else right_val
            for i in range(start, end):
                out[row + i] = fillv
    return out


def predict_consistent(net, left, right, W, H, work_width=None,
                       tol=1.5, verbose=True):
    """Disparity with a left-right consistency check applied.

    Returns (disp, valid_fraction). Costs two forward passes.
    """
    if work_width is None:
        d_l, _, Ww, Hw = predict_auto(net, left, right, W, H, verbose=verbose)
    else:
        d_l, _, Ww, Hw = predict_at_scale(net, left, right, W, H,
                                          work_width, verbose)

    # Right-reference map, via mirroring.
    ml = mirror_h(left, 1, W, H)
    mr = mirror_h(right, 1, W, H)
    d_r_mirrored, _, _, _ = predict_at_scale(net, mr, ml, W, H, Ww,
                                             verbose=False)
    d_r = mirror_h(d_r_mirrored, 1, W, H)

    valid = lr_consistency(d_l, d_r, W, H, tol)
    kept = 0
    for v in valid:
        if v:
            kept += 1
    frac = kept / float(W * H)
    if verbose:
        print('  left-right consistency: %.1f%% of pixels agree within '
              '%.1f px' % (100.0 * frac, tol))
    return fill_invalid(d_l, valid, W, H), frac


# =====================================================================
#  SECTION 25 : FULL-RESOLUTION REFINEMENT AND PLANES
#
#  The network runs on a shrunken pair and works at a quarter of that
#  size, so one of its pixels covers about two pixels of the input. It
#  finds the right surface but cannot place it precisely, and in 3D that
#  imprecision shows up as relief on flat walls.
#
#  The fix keeps the network as the coarse answer and matches again at
#  full resolution, examining only 2R+1 disparities around it:
#
#    1. census signatures (9x7 window) compare local intensity order,
#       which survives the exposure differences between two cameras;
#    2. semi-global matching (Hirschmuller 2008) along 8 directions adds
#       a small penalty P1 for a 1 px step and a large P2 for a jump,
#       with the band re-centred on each pixel's own coarse estimate;
#    3. a parabola through the winning cost and its neighbours gives
#       the sub-pixel value;
#    4. the same is done with the right image as reference, and pixels
#       where the two disagree are either occlusions (filled from the
#       farther neighbour, since what is hidden is background) or
#       mismatches (given back the network's value);
#    5. the left image is cut into SLIC superpixels, and each segment
#       that is well described by a slanted plane d = ax + by + c has
#       its pixels snapped onto that plane;
#    6. for display, a thin background layer is continued behind every
#       foreground edge, so a rotated view shows background rather than
#       a hole. That layer is completion, not measurement, and is kept
#       out of every accuracy figure.
#
#  A winner at the edge of the band means the true value may lie outside
#  it, so such pixels keep the network's estimate. Measured on the five
#  validation scenes, this took end-point error from 1.65 to 0.89 px and
#  3D surface-normal error from 38 to 23 degrees.
# =====================================================================

def census(img, W, H, rx=2, ry=2):
    """Census signature per pixel: bit set where neighbour < centre."""
    out = [0] * (W * H)
    offs = []
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if dx or dy:
                offs.append((dx, dy))
    for y in range(H):
        for x in range(W):
            c = img[y * W + x]
            s = 0
            for dx, dy in offs:
                xx = x + dx
                yy = y + dy
                if xx < 0:
                    xx = 0
                elif xx >= W:
                    xx = W - 1
                if yy < 0:
                    yy = 0
                elif yy >= H:
                    yy = H - 1
                s = s + s
                if img[yy * W + xx] < c:
                    s += 1
            out[y * W + x] = s
    return out


_POP8 = [bin(i).count('1') for i in range(256)]


def _popcount(v):
    n = 0
    while v:
        n += _POP8[v & 255]
        v >>= 8
    return n


def band_costs(cl, cr, gl, gr, base, W, H, R, ad_weight, ad_trunc, invalid,
               prior=None, prior_w=0.0, prior_t=4.0):
    """Matching cost for d = base[p] + k - R, k in 0..2R."""
    K = 2 * R + 1
    C = [invalid] * (W * H * K)
    for y in range(H):
        row = y * W
        for x in range(W):
            p = row + x
            b = base[p] - R
            a = cl[p]
            il = gl[p]
            o = p * K
            for k in range(K):
                xr = x - (b + k)
                if b + k < 0 or xr < 0 or xr >= W:
                    continue
                q = row + xr
                ad = il - gr[q]
                if ad < 0:
                    ad = -ad
                ad = ad * 255.0
                if ad > ad_trunc:
                    ad = ad_trunc
                c = _popcount(a ^ cr[q]) + ad_weight * ad
                if prior_w > 0.0:
                    e = b + k - prior[p]
                    if e < 0:
                        e = -e
                    if e > prior_t:
                        e = prior_t
                    c += prior_w * e
                C[o + k] = c
    return C


def _aggregate_path(C, base, W, H, K, P1, P2, dx, dy, S):
    """One SGM path, added into S."""
    R = K // 2
    N = W * H
    L = [0.0] * (N * K)
    mins = [0.0] * N
    ys = range(H) if dy >= 0 else range(H - 1, -1, -1)
    xs = list(range(W)) if dx >= 0 else list(range(W - 1, -1, -1))
    for y in ys:
        for x in xs:
            p = y * W + x
            o = p * K
            qx = x - dx
            qy = y - dy
            if qx < 0 or qx >= W or qy < 0 or qy >= H:
                m = 1.0e30
                for k in range(K):
                    v = C[o + k]
                    L[o + k] = v
                    if v < m:
                        m = v
                mins[p] = m
                continue
            q = qy * W + qx
            oq = q * K
            mq = mins[q]
            shift = base[p] - base[q]
            m = 1.0e30
            for k in range(K):
                kq = k + shift
                best = mq + P2
                if 0 <= kq < K:
                    v = L[oq + kq]
                    if v < best:
                        best = v
                if 0 <= kq - 1 < K:
                    v = L[oq + kq - 1] + P1
                    if v < best:
                        best = v
                if 0 <= kq + 1 < K:
                    v = L[oq + kq + 1] + P1
                    if v < best:
                        best = v
                v = C[o + k] + best - mq
                L[o + k] = v
                if v < m:
                    m = v
            mins[p] = m
    for i in range(N * K):
        S[i] += L[i]


def refine_fullres(left, right, disp0, W, H, R=3, P1=1.0, P2=12.0,
                   ad_weight=0.0, ad_trunc=20.0, paths=8, uniq=0.0,
                   rx=2, ry=2, prior_w=0.0, prior_t=4.0, return_conf=False):
    """Refine a coarse disparity map at full resolution.

    Only 2R+1 disparities around the coarse estimate are examined per
    pixel, so this costs a few seconds rather than a full search. The
    coarse map says which surface a pixel is on; matching at full
    resolution says exactly where.
    """
    K = 2 * R + 1
    base = [0] * (W * H)
    for i in range(W * H):
        v = int(disp0[i] + 0.5)
        base[i] = v if v > 0 else 0
    nbits = (2 * rx + 1) * (2 * ry + 1) - 1
    cl = census(left, W, H, rx, ry)
    cr = census(right, W, H, rx, ry)
    invalid = nbits + ad_weight * ad_trunc + prior_w * prior_t + 4.0
    C = band_costs(cl, cr, left, right, base, W, H, R, ad_weight, ad_trunc,
                   invalid, disp0, prior_w, prior_t)
    S = [0.0] * (W * H * K)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if paths == 8:
        dirs += [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for dx, dy in dirs:
        _aggregate_path(C, base, W, H, K, P1, P2, dx, dy, S)
    out = [0.0] * (W * H)
    conf = [0.0] * (W * H)
    for p in range(W * H):
        o = p * K
        bk = 0
        bv = S[o]
        for k in range(1, K):
            if S[o + k] < bv:
                bv = S[o + k]
                bk = k
        if bk == 0 or bk == K - 1:
            out[p] = disp0[p]
            conf[p] = 0.0
            continue
        # second best, excluding the winner's neighbours
        sv = 1.0e30
        for k in range(K):
            if k < bk - 1 or k > bk + 1:
                if S[o + k] < sv:
                    sv = S[o + k]
        c = (sv - bv) / (bv + 1.0e-6) if sv < 1.0e29 else 1.0
        conf[p] = c
        if c < uniq:
            out[p] = disp0[p]
            continue
        a = S[o + bk - 1]
        b = S[o + bk + 1]
        den = a - 2.0 * bv + b
        off = 0.0
        if den > 1.0e-9:
            off = 0.5 * (a - b) / den
        out[p] = base[p] - R + bk + off
    if return_conf:
        return out, conf
    return out


def _mirror1(img, W, H):
    out = [0.0] * (W * H)
    for y in range(H):
        r = y * W
        for x in range(W):
            out[r + x] = img[r + W - 1 - x]
    return out


def left_to_right(dl, W, H):
    """Forward-warp a left disparity map into the right view (nearest wins),
    filling holes with the farther neighbour."""
    dr = [-1.0] * (W * H)
    for y in range(H):
        r = y * W
        for x in range(W):
            d = dl[r + x]
            xr = int(x - d + 0.5)
            if 0 <= xr < W and d > dr[r + xr]:
                dr[r + xr] = d
        for x in range(W):
            if dr[r + x] < 0.0:
                a = x - 1
                while a >= 0 and dr[r + a] < 0.0:
                    a -= 1
                b = x + 1
                while b < W and dr[r + b] < 0.0:
                    b += 1
                va = dr[r + a] if a >= 0 else 1.0e9
                vb = dr[r + b] if b < W else 1.0e9
                v = va if va < vb else vb
                dr[r + x] = v if v < 1.0e8 else 0.0
    return dr


def lr_refine(left, right, disp0, W, H, tol=1.0, **kw):
    """Refine both views and keep what agrees.

    A pixel whose left and right estimates disagree is either occluded
    (the right view sees something nearer there) or a mismatch. Occluded
    pixels take the farther of their valid neighbours on the same row, since
    what hides behind an object is background. Mismatches fall back to the
    network's estimate.
    Returns (disparity, status) with status 0 ok, 1 occluded, 2 mismatch.
    """
    dl = refine_fullres(left, right, disp0, W, H, **kw)
    d0r = left_to_right(disp0, W, H)
    drm = refine_fullres(_mirror1(right, W, H), _mirror1(left, W, H),
                         _mirror1(d0r, W, H), W, H, **kw)
    dr = _mirror1(drm, W, H)
    status = [0] * (W * H)
    out = dl[:]
    for y in range(H):
        r = y * W
        for x in range(W):
            d = dl[r + x]
            xr = int(x - d + 0.5)
            if xr < 0 or xr >= W:
                status[r + x] = 1
                continue
            e = dr[r + xr]
            if abs(d - e) <= tol:
                continue
            status[r + x] = 1 if e > d else 2
    for i in range(W * H):
        if status[i] == 2:
            out[i] = disp0[i]
    # occlusions: farther valid neighbour on the row
    for y in range(H):
        r = y * W
        for x in range(W):
            if status[r + x] != 1:
                continue
            a = x - 1
            while a >= 0 and status[r + a] == 1:
                a -= 1
            b = x + 1
            while b < W and status[r + b] == 1:
                b += 1
            va = out[r + a] if a >= 0 else 1.0e9
            vb = out[r + b] if b < W else 1.0e9
            v = va if va < vb else vb
            out[r + x] = v if v < 1.0e8 else disp0[r + x]
    return out, status


def slic(rgb, W, H, step=12, compactness=10.0, iters=5):
    """SLIC superpixels on RGB (0..1). Returns a label per pixel."""
    cs = []
    y = step // 2
    while y < H:
        x = step // 2
        while x < W:
            i = y * W + x
            cs.append([rgb[3 * i] * 255.0, rgb[3 * i + 1] * 255.0,
                       rgb[3 * i + 2] * 255.0, float(x), float(y)])
            x += step
        y += step
    N = W * H
    lab = [-1] * N
    m2 = (compactness / step) ** 2
    for _ in range(iters):
        dist = [1.0e30] * N
        for ci in range(len(cs)):
            cr, cg, cb, cx, cy = cs[ci]
            x0 = int(cx) - step
            x1 = int(cx) + step
            y0 = int(cy) - step
            y1 = int(cy) + step
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 >= W:
                x1 = W - 1
            if y1 >= H:
                y1 = H - 1
            for yy in range(y0, y1 + 1):
                r = yy * W
                dy = yy - cy
                for xx in range(x0, x1 + 1):
                    i = r + xx
                    a = rgb[3 * i] * 255.0 - cr
                    b = rgb[3 * i + 1] * 255.0 - cg
                    c = rgb[3 * i + 2] * 255.0 - cb
                    dx = xx - cx
                    dd = a * a + b * b + c * c + m2 * (dx * dx + dy * dy)
                    if dd < dist[i]:
                        dist[i] = dd
                        lab[i] = ci
        acc = [[0.0, 0.0, 0.0, 0.0, 0.0, 0] for _ in cs]
        for i in range(N):
            l = lab[i]
            if l < 0:
                continue
            a = acc[l]
            a[0] += rgb[3 * i]
            a[1] += rgb[3 * i + 1]
            a[2] += rgb[3 * i + 2]
            a[3] += i % W
            a[4] += i // W
            a[5] += 1
        for ci in range(len(cs)):
            a = acc[ci]
            if a[5]:
                n = float(a[5])
                cs[ci] = [a[0] / n * 255.0, a[1] / n * 255.0, a[2] / n * 255.0,
                          a[3] / n, a[4] / n]
    return lab, len(cs)


def _solve3(A, b):
    (a, bb, c), (d, e, f), (g, h, k) = A
    det = a * (e * k - f * h) - bb * (d * k - f * g) + c * (d * h - e * g)
    if abs(det) < 1.0e-9:
        return None
    x = (b[0] * (e * k - f * h) - bb * (b[1] * k - f * b[2]) +
         c * (b[1] * h - e * b[2])) / det
    y = (a * (b[1] * k - f * b[2]) - b[0] * (d * k - f * g) +
         c * (d * b[2] - b[1] * g)) / det
    z = (a * (e * b[2] - b[1] * h) - bb * (d * b[2] - b[1] * g) +
         b[0] * (d * h - e * g)) / det
    return (x, y, z)


def _lsq_plane(xs, ys, ds):
    n = float(len(xs))
    # explicit loops: sum() of floats is compensated on CPython 3.12+
    # and naive elsewhere, and the result must not depend on that
    sx = 0.0
    for v in xs:
        sx += v
    sy = 0.0
    for v in ys:
        sy += v
    sd = 0.0
    for v in ds:
        sd += v
    mx = sx / n; my = sy / n; md = sd / n
    sxx = sxy = syy = sxd = syd = 0.0
    for x, y, d in zip(xs, ys, ds):
        u = x - mx
        v = y - my
        w = d - md
        sxx += u * u
        sxy += u * v
        syy += v * v
        sxd += u * w
        syd += v * w
    det = sxx * syy - sxy * sxy
    if abs(det) < 1.0e-9:
        return (0.0, 0.0, md)
    a = (sxd * syy - syd * sxy) / det
    b = (syd * sxx - sxd * sxy) / det
    return (a, b, md - a * mx - b * my)


def plane_snap(disp, rgb, W, H, valid=None, step=12, thr=1.0, min_inl=0.6,
               trials=40, seed=12345):
    """Fit a slanted plane d = a x + b y + c to each colour segment and snap
    the pixels that lie on it.

    Walls, tables and box faces are planes, and a plane in space is an
    affine function of the image coordinates in disparity. Pixel noise in
    the disparity turns into relief in 3D; snapping to a fitted plane
    removes it. A segment where too few pixels agree with any plane (cloth,
    foliage) is left untouched.
    """
    lab, nseg = slic(rgb, W, H, step)
    groups = [[] for _ in range(nseg)]
    for i in range(W * H):
        if lab[i] >= 0:
            groups[lab[i]].append(i)
    out = disp[:]
    s = seed
    snapped = 0
    for g in groups:
        pts = [i for i in g if valid is None or valid[i]]
        n = len(pts)
        if n < 12:
            continue
        best = None
        bestc = -1
        for _ in range(trials):
            tri = []
            for __ in range(3):
                s ^= (s << 13) & 0xFFFFFFFF
                s ^= s >> 17
                s ^= (s << 5) & 0xFFFFFFFF
                tri.append(pts[s % n])
            A = [[float(i % W), float(i // W), 1.0] for i in tri]
            pl = _solve3(A, [disp[i] for i in tri])
            if pl is None:
                continue
            a, b, c = pl
            cnt = 0
            for i in pts:
                if abs(a * (i % W) + b * (i // W) + c - disp[i]) <= thr:
                    cnt += 1
            if cnt > bestc:
                bestc = cnt
                best = pl
        if best is None or bestc < min_inl * n:
            continue
        a, b, c = best
        inl = [i for i in pts
               if abs(a * (i % W) + b * (i // W) + c - disp[i]) <= thr]
        a, b, c = _lsq_plane([float(i % W) for i in inl],
                             [float(i // W) for i in inl],
                             [disp[i] for i in inl])
        for i in g:
            v = a * (i % W) + b * (i // W) + c
            if abs(v - disp[i]) <= thr or (valid is not None and not valid[i]):
                out[i] = v if v > 0.0 else 0.0
                snapped += 1
    return out


def min_filter_arg(v, W, H, r, rv=None):
    """Minimum filter (horizontal radius r, vertical radius rv, default r)
    that also returns, per pixel, the index of the pixel the minimum came
    from. Separable: rows, then columns."""
    if rv is None:
        rv = r
    N = W * H
    rowv = [0.0] * N
    ri = [0] * N
    for y in range(H):
        row = y * W
        for x in range(W):
            a = x - r if x - r > 0 else 0
            b = x + r if x + r < W - 1 else W - 1
            bi = row + a
            bv = v[bi]
            for xx in range(a + 1, b + 1):
                if v[row + xx] < bv:
                    bv = v[row + xx]
                    bi = row + xx
            rowv[row + x] = bv
            ri[row + x] = bi
    if rv == 0:
        return rowv, ri
    out = [0.0] * N
    oi = [0] * N
    for x in range(W):
        for y in range(H):
            a = y - rv if y - rv > 0 else 0
            b = y + rv if y + rv < H - 1 else H - 1
            bi = a * W + x
            bv = rowv[bi]
            for yy in range(a + 1, b + 1):
                j = yy * W + x
                if rowv[j] < bv:
                    bv = rowv[j]
                    bi = j
            out[y * W + x] = bv
            oi[y * W + x] = ri[bi]
    return out, oi


def smooth_background(m, disp, W, H, r):
    """Box-average the min-filtered background so it forms ramps rather
    than flat steps, never letting it come in front of the pixel's own
    surface."""
    N = W * H
    t = [0.0] * N
    for y in range(H):
        row = y * W
        for x in range(W):
            a = x - r if x - r > 0 else 0
            b = x + r if x + r < W - 1 else W - 1
            s = 0.0
            for xx in range(a, b + 1):
                s += m[row + xx]
            t[row + x] = s / (b - a + 1)
    out = [0.0] * N
    for x in range(W):
        for y in range(H):
            a = y - r if y - r > 0 else 0
            b = y + r if y + r < H - 1 else H - 1
            s = 0.0
            for yy in range(a, b + 1):
                s += t[yy * W + x]
            v = s / (b - a + 1)
            i = y * W + x
            out[i] = v if v < disp[i] else disp[i]
    return out


def background_region(disp, W, H, radius=None, jump_ratio=0.15,
                      min_jump=2.0):
    """Pixels on the foreground side of a depth edge, plus 2 px, with the
    background disparity (smoothed) and the pixel it was copied from."""
    # Horizontal only: stereo occlusions run along rows, and turning the
    # view left or right is what exposes them. A square window also
    # reached above and below objects and hung large sheets off them.
    if radius is None:
        radius = int(W / 16) if W >= 64 else 4
    m, src = min_filter_arg(disp, W, H, radius, 0)
    m = smooth_background(m, disp, W, H, radius // 2 if radius >= 2 else 1)
    N = W * H
    mask = [False] * N
    for i in range(N):
        d = disp[i]
        thr = jump_ratio * d
        if thr < min_jump:
            thr = min_jump
        if d - m[i] > thr:
            mask[i] = True
    # grow by 2 px so the layer overlaps the real background it continues
    for _ in range(2):
        grown = mask[:]
        for y in range(H):
            row = y * W
            for x in range(W):
                if mask[row + x]:
                    continue
                if ((x > 0 and mask[row + x - 1]) or
                        (x < W - 1 and mask[row + x + 1]) or
                        (y > 0 and mask[row - W + x]) or
                        (y < H - 1 and mask[row + W + x])):
                    grown[row + x] = True
        mask = grown
    return mask, m, src


def background_layer(disp, rgb, W, H, cam, n_points, radius=None,
                     jump_ratio=0.15, min_jump=2.0, min_disp=0.5):
    """A second surface behind every foreground edge.

    Seen from the original camera, a foreground object hides the
    background behind its silhouette. Rotate the view and that hidden
    strip appears as a hole, because the depth map holds one surface per
    pixel. This builds a thin extra layer there, continuing the nearby
    background (the farthest disparity within `radius`) behind the
    object and colouring it from the background pixel it was copied
    from. It is completion, not measurement: the stereo pair never saw
    these surfaces.

    Returns (points2d, tris, pts3d, rgb) for the layer, in the same
    per-pixel layout as the main mesh.
    """
    mask, m, src = background_region(disp, W, H, radius, jump_ratio,
                                     min_jump)
    N = W * H
    count = sum(1 for b in mask if b)
    if count < 12:
        return [], [], [None] * N, [0.0] * (3 * N)
    lrgb = [0.0] * (3 * N)
    pts3d = [None] * N
    for i in range(N):
        if not mask[i]:
            continue
        s = src[i]
        lrgb[3 * i] = rgb[3 * s]
        lrgb[3 * i + 1] = rgb[3 * s + 1]
        lrgb[3 * i + 2] = rgb[3 * s + 2]
        d = m[i]
        if d >= min_disp:
            pts3d[i] = cam.unproject(i % W, i // W, cam.depth_from_disparity(d))
    # vertices: a grid over the masked area at the main mesh's density
    step = fsqrt(N / float(n_points if n_points > 0 else 1))
    s = int(step + 0.5)
    if s < 2:
        s = 2
    pts = []
    for y in range(0, H, s):
        for x in range(0, W, s):
            if mask[y * W + x]:
                pts.append((x, y))
    # and the outline, so the layer keeps its shape
    for y in range(H):
        row = y * W
        for x in range(W):
            i = row + x
            if not mask[i]:
                continue
            edge = (x == 0 or x == W - 1 or y == 0 or y == H - 1 or
                    not mask[i - 1] or not mask[i + 1] or
                    not mask[i - W] or not mask[i + W])
            if edge and (x % 2 == 0) and (y % 2 == 0):
                pts.append((x, y))
    if len(pts) < 3:
        return [], [], pts3d, lrgb
    tris = delaunay(pts)
    keep = []
    for (a, b, c) in tris:
        cx = int((pts[a][0] + pts[b][0] + pts[c][0]) / 3.0)
        cy = int((pts[a][1] + pts[b][1] + pts[c][1]) / 3.0)
        if mask[cy * W + cx]:
            keep.append((a, b, c))
    keep = cull_triangles(pts, keep, pts3d, W)
    return pts, keep, pts3d, lrgb


def merge_layer(points2d, tris, pts3d, rgb, layer, W, H):
    """Place the main mesh and the background layer side by side on a
    canvas 2W wide, so every function that looks vertices up per pixel
    works unchanged. Layer vertex (x, y) becomes (x + W, y)."""
    lp, lt, l3, lrgb = layer
    W2 = 2 * W
    p3 = [None] * (W2 * H)
    c = [0.0] * (3 * W2 * H)
    for y in range(H):
        for x in range(W):
            i = y * W + x
            j = y * W2 + x
            k = j + W
            p3[j] = pts3d[i]
            p3[k] = l3[i]
            c[3 * j] = rgb[3 * i]
            c[3 * j + 1] = rgb[3 * i + 1]
            c[3 * j + 2] = rgb[3 * i + 2]
            c[3 * k] = lrgb[3 * i]
            c[3 * k + 1] = lrgb[3 * i + 1]
            c[3 * k + 2] = lrgb[3 * i + 2]
    n = len(points2d)
    pts = list(points2d) + [(x + W, y) for (x, y) in lp]
    tr = list(tris) + [(a + n, b + n, c2 + n) for (a, b, c2) in lt]
    return pts, tr, p3, c, W2


def grid_mesh(disp, W, H, cam, step=2, fill=True, jump=1.5, jump_rel=0.06,
              min_disp=None):
    """Dense grid mesh for the viewer: a vertex every `step` pixels, two
    triangles per cell, and texture coordinates equal to the pixel
    position, so the photograph is drawn onto the surface pixel for
    pixel instead of one flat colour per triangle.

    Cells are dropped where disparity jumps (an occlusion edge), where
    depth is beyond the far clamp, and where x < d: the right camera never
    saw those pixels, so nothing was measured there. With fill=True a
    second grid continues the background behind foreground edges
    (section 25), textured from the pixel each value was copied from.
    Same rules as buildGridMesh in web/recon.js.
    """
    if min_disp is None:
        min_disp = robust_min_disparity(disp)
    xs = list(range(0, W, step))
    if xs[-1] != W - 1:
        xs.append(W - 1)
    ys = list(range(0, H, step))
    if ys[-1] != H - 1:
        ys.append(H - 1)
    nx = len(xs)
    ny = len(ys)
    pos = []
    uv = []
    flag = []
    dv = []
    idx = []

    def build(value_at, uv_at, is_fill, region):
        vid = [-1] * (nx * ny)
        for j in range(ny):
            for i in range(nx):
                x = xs[i]
                y = ys[j]
                p = y * W + x
                if region is not None and not region[p]:
                    continue
                d = value_at(p)
                if not d >= min_disp:
                    continue
                if not is_fill and x - d < 0:
                    continue
                Z = cam.depth_from_disparity(d)
                if Z <= 0.0:
                    continue
                P = cam.unproject(x, y, Z)
                vid[j * nx + i] = len(dv)
                pos.extend((P[0], P[1], P[2]))
                u, v = uv_at(p, x, y)
                uv.extend((u, v))
                flag.append(1 if is_fill else 0)
                dv.append(d)

        def ok(a, b, c):
            if a < 0 or b < 0 or c < 0:
                return False
            da = dv[a]
            db = dv[b]
            dc = dv[c]
            lo = fmin(da, fmin(db, dc))
            hi = fmax(da, fmax(db, dc))
            return hi - lo <= fmax(jump, jump_rel * hi)
        n = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = vid[j * nx + i]
                b = vid[j * nx + i + 1]
                c = vid[(j + 1) * nx + i]
                e = vid[(j + 1) * nx + i + 1]
                if ok(a, b, c):
                    idx.extend((a, c, b))
                    n += 1
                if ok(b, c, e):
                    idx.extend((b, c, e))
                    n += 1
        return n

    main = build(lambda p: disp[p],
                 lambda p, x, y: ((x + 0.5) / W, (y + 0.5) / H), False, None)
    nfill = 0
    if fill:
        mask, m, src = background_region(disp, W, H)
        nfill = build(lambda p: m[p],
                      lambda p, x, y: ((src[p] % W + 0.5) / W,
                                       (src[p] // W + 0.5) / H),
                      True, mask)
    zs = sorted(pos[3 * k + 2] for k in range(len(flag)) if not flag[k])
    pivot = zs[len(zs) // 2] if zs else 1.0
    return {'pos': pos, 'uv': uv, 'fill': flag, 'disp': dv, 'idx': idx,
            'main_tris': main, 'fill_tris': nfill, 'pivot': pivot,
            'fx': cam.fx, 'cx': cam.cx, 'cy': cam.cy, 'W': W, 'H': H}


def refine_disparity(left, right, rgb, disp0, W, H, R=10, P1=12.0, P2=150.0,
                     rx=4, ry=3, tol=2.0, planes=True, plane_step=12,
                     plane_thr=1.0, plane_min_inl=0.4, verbose=False):
    """Network estimate -> full-resolution band SGM (both views) -> planes."""
    d, status = lr_refine(left, right, disp0, W, H, tol=tol, R=R, P1=P1,
                          P2=P2, rx=rx, ry=ry)
    if verbose:
        n = float(W * H)
        print('  full-res refinement: %.1f%% occluded, %.1f%% mismatched'
              % (100.0 * sum(1 for s in status if s == 1) / n,
                 100.0 * sum(1 for s in status if s == 2) / n))
    if planes and rgb is not None:
        d = plane_snap(d, rgb, W, H, valid=[s == 0 for s in status],
                       step=plane_step, thr=plane_thr,
                       min_inl=plane_min_inl)
    # One 3x3 median removes single-pixel spikes, each of which would
    # otherwise lift a mesh vertex into a needle. On validation it also
    # lowered EPE (0.89 -> 0.86), AbsRel and normal error.
    d = median_filter(d, W, H, radius=1, passes=1)
    return d, status


# =====================================================================
#  SECTION 17 : END-TO-END PIPELINE AND DEMO
# =====================================================================

def predict(net, left, right, W, H):
    """Run the network on a rectified pair and return full-resolution
    disparity. The model is fully convolutional, so any size works;
    cost grows roughly linearly in pixel count."""
    guide_half, H2, W2 = avgpool2(left, 1, H, W)
    d0, d1, d2 = net.forward(left, right, H, W, guide_half, left)
    return d2


def run_pipeline(net, sample, cam, out_prefix, n_points=480,
                 refine_iters=90, n_views=5, verbose=True):
    """Stereo pair -> disparity -> mesh -> Gaussians -> novel views."""
    W, H = sample['W'], sample['H']

    if verbose:
        print('STAGE 1  disparity estimation')
    disp = predict(net, sample['left'], sample['right'], W, H)

    if sample.get('disp') is not None:
        epe, b1, b3 = metrics(disp, sample['disp'], sample['valid'])
        if verbose:
            print('  EPE %.3f px   bad1.0 %.1f%%   bad3.0 %.1f%%'
                  % (epe, b1, b3))
    write_pgm(out_prefix + '_disparity.pgm', disp, W, H)

    if verbose:
        print('STAGE 2  3D reconstruction')
    rgb = sample['left_rgb']
    if rgb is None:
        rgb = [0.0] * (3 * W * H)
        for i in range(W * H):
            rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = sample['left'][i]

    res = reconstruct(disp, rgb, W, H, cam, valid=None,
                      n_points=n_points, refine_iters=refine_iters,
                      out_prefix=out_prefix, n_views=n_views,
                      verbose=verbose)
    res['disp'] = disp
    return res


def run_demo(weights='stereonet_weights.txt', out_prefix='demo',
             seed=20260920, train_if_missing=True, train_iters=400):
    """The whole thing, start to finish."""
    rng = Rng(seed)
    net = StereoNet(rng, feat_C=8, cost_C=4, D=8, max_disp=32)

    loaded = False
    try:
        load_weights(net, weights)
        loaded = True
        print('loaded weights from %s' % weights)
    except Exception:
        print('no usable checkpoint at %s' % weights)

    if not loaded and train_if_missing:
        print('training from scratch (%d iterations)' % train_iters)
        train(net, rng, iters=train_iters, ckpt=weights)

    W = H = 64
    sample = make_stereogram(Rng(seed + 1), W, H, max_disp=16, n_objects=3)
    cam = Camera.synthetic(W, H, baseline=0.10)

    print('')
    print('camera  fx %.1f px   baseline %.2f m   '
          'depth range %.2f m to %.2f m'
          % (cam.fx, cam.baseline,
             cam.depth_from_disparity(16.0),
             cam.depth_from_disparity(2.0)))
    print('')

    res = run_pipeline(net, sample, cam, out_prefix)

    print('')
    print('GROUND TRUTH DISPARITY')
    print(ascii_view(sample['disp'], W, H, 60))
    print('')
    print('PREDICTED DISPARITY')
    print(ascii_view(res['disp'], W, H, 60))
    print('')
    print('wrote:')
    print('  %s_disparity.pgm        depth map, viewable anywhere'
          % out_prefix)
    print('  %s_mesh.obj             Delaunay mesh with vertex colour'
          % out_prefix)
    print('  %s_render_original.ppm  Gaussians re-rendered from the '
          'input viewpoint' % out_prefix)
    print('  %s_novel_0..4.ppm       viewpoints the camera never '
          'occupied' % out_prefix)
    return res


if __name__ == '__main__':
    run_demo()
