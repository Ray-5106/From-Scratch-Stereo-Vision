"""Compare the exact triangle covariance (Eq. 3) with the GaMeS face
parameterisation (Waczynska et al., 2024, arXiv:2402.01459), as we read its
equations: r2 = unit(v1 - m), r3 = Gram-Schmidt(v2 - m), s2 = |v1 - m|,
s3 = <v2 - m, r3>, normal scale = epsilon (omitted in 2D).
Harness script; numpy allowed."""
import numpy as np
rng = np.random.default_rng(0)

def exact(P):
    u, v = P[1] - P[0], P[2] - P[0]
    return (np.outer(u, u) + np.outer(v, v)) / 18 - (np.outer(u, v) + np.outer(v, u)) / 36

def games(P):
    m = P.mean(0)
    r2 = P[0] - m; s2 = np.linalg.norm(r2); r2 /= s2
    w = P[1] - m; w = w - (w @ r2) * r2; s3 = np.linalg.norm(w); r3 = w / s3
    return s2 ** 2 * np.outer(r2, r2) + s3 ** 2 * np.outer(r3, r3)

angles, order_change = [], []
for _ in range(10000):
    P = rng.normal(size=(3, 2))
    ea, va = np.linalg.eigh(exact(P)); eg, vg = np.linalg.eigh(games(P))
    angles.append(np.degrees(np.arccos(min(1, abs(va[:, 1] @ vg[:, 1])))))
    _, vg2 = np.linalg.eigh(games(P[[1, 2, 0]]))
    order_change.append(np.degrees(np.arccos(min(1, abs(vg[:, 1] @ vg2[:, 1])))))
    _, va2 = np.linalg.eigh(exact(P[[1, 2, 0]]))
    assert abs(abs(va[:, 1] @ va2[:, 1]) - 1) < 1e-9          # Eq. 3 ignores vertex order
angles, order_change = np.array(angles), np.array(order_change)
print("major-axis difference, exact vs GaMeS: median %.1f deg, 90th percentile %.1f deg"
      % (np.median(angles), np.percentile(angles, 90)))
print("GaMeS major axis change under vertex reordering: median %.1f deg, 90th percentile %.1f deg"
      % (np.median(order_change), np.percentile(order_change, 90)))
print("exact covariance: unchanged under reordering (checked on all 10,000 triangles)")
