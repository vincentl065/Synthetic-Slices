"""NMS 2 line extraction (v0.2): one continuous line per sheet, no branches.

Instead of "threshold + skeleton" (which turns every bridge between two sheets into a junction), a pixel is kept only if the probability
map is a local maximum ACROSS the sheet:
  - the large-scale sheet normal comes from the structure tensor of the map (smoothing SO = 8 px);
  - the maximum is searched along the stroke's own normal (Hessian, sigma SH = 1.5 px), and the stroke is kept only if that normal is within
    THETA = 45 deg of the large-scale sheet normal: a bridge, which runs ACROSS the sheets (~90 deg), is rejected, a sheet turning by 20-30
    deg is not;
  - no curvature threshold (kc = 0: any threshold cost recall on low-confidence slices without removing a single branch);
  - hysteresis at 1 x and 0.5 x the Otsu threshold, thinning;
  - spurs < 10 px pruned, any remaining junction pixel removed (safety net), pieces < 5 px removed, and gaps <= 6 px closed between two
    line ends that face each other (never towards the middle of a line, so no junction is created).
Settings (SETTINGS) were frozen on a tuning set (held-out real tiles + held-out synthetic tiles), never on the test slices.

    lines = extract_lines(prob)            # prob (H, W) float, any range -> uint8 (H, W), 255 on the lines
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
K8 = np.ones((3, 3), np.uint8)
K8[1, 1] = 0
C8 = np.ones((3, 3))
SETTINGS = dict(so=8.0, sh=1.5, theta=45.0, kc=0.0, hi=1.0, lo=0.5, loc=0.0, ep=10, lmin=5, gap=6.0)

def voisins(S):
    return ndi.convolve(S.astype(np.uint8), K8, mode='constant') * S

def elaguer(S, lmin):
    for _ in range(3):
        nb = voisins(S)
        J = S & (nb >= 3)
        if not J.any():
            break
        seg = S & ~ndi.binary_dilation(J, C8)
        lab, n = ndi.label(seg, C8)
        if n == 0:
            break
        idx = np.arange(1, n + 1)
        t = ndi.sum(np.ones(lab.shape, np.uint8), lab, idx)
        E = S & (nb == 1)
        aj = ndi.maximum(ndi.binary_dilation(J, np.ones((5, 5))).astype(np.uint8), lab, idx)
        ae = ndi.maximum(E.astype(np.uint8), lab, idx)
        ret = np.nonzero((t < lmin) & (aj > 0) & (ae > 0))[0] + 1
        if not len(ret):
            break
        S = S.copy()
        S[np.isin(lab, ret)] = False
        S = skeletonize(S)
    return S

def couper_jonctions(S):
    nb = voisins(S)
    J = S & (nb >= 3)
    if J.any():
        S = S & ~J
    return S

def combler(S, m, gap, tx, ty):
    from scipy.spatial import cKDTree
    from skimage.draw import line as dline
    nb = voisins(S)
    E = np.argwhere(S & (nb == 1))
    if len(E) < 2:
        return S
    lab, _ = ndi.label(S, C8)
    T = cKDTree(E)
    S = S.copy()
    pris = set()
    for i, j in sorted(T.query_pairs(gap), key=lambda p: np.hypot(*E[p[0]] - E[p[1]])):
        if i in pris or j in pris or lab[tuple(E[i])] == lab[tuple(E[j])]:
            continue
        d = (E[j] - E[i]).astype(float)
        L = np.hypot(*d)
        d /= L + 1e-09
        ok = True
        for k in (i, j):
            y, x = E[k]
            c = abs(d[0] * ty[y, x] + d[1] * tx[y, x])
            if c < 0.82:
                ok = False
        if not ok:
            continue
        rr, cc = dline(int(E[i][0]), int(E[i][1]), int(E[j][0]), int(E[j][1]))
        if not m[rr, cc].all() or S[rr[1:-1], cc[1:-1]].any():
            continue
        S[rr, cc] = True
        pris.add(i)
        pris.add(j)
    return S

def nms2(q, m, so=8.0, sh=1.5, theta=45.0, kc=0.05, hi=1.0, lo=0.5, loc=0.0, ep=10, lmin=5, gap=0.0):
    q = q.astype(np.float32)
    if not m.any():
        return np.zeros(q.shape, np.uint8)
    q = (q - q[m].min()) / (q[m].max() - q[m].min() + 1e-06)
    qs = gaussian_filter(q, 1.0)
    gx, gy = (gaussian_filter(q, 1.5, order=(0, 1)), gaussian_filter(q, 1.5, order=(1, 0)))
    J = [gaussian_filter(a, so) for a in (gx * gx, gy * gy, gx * gy)]
    phi = 0.5 * np.arctan2(2 * J[2], J[0] - J[1])
    nx, ny = (np.cos(phi), np.sin(phi))
    if theta > 0:
        hxx, hyy, hxy = (gaussian_filter(q, sh, order=(0, 2)), gaussian_filter(q, sh, order=(2, 0)), gaussian_filter(q, sh, order=(1, 1)))
        th = 0.5 * np.arctan2(2 * hxy, hxx - hyy)
        vx, vy = (-np.sin(th), np.cos(th))
        porte = np.abs(vx * nx + vy * ny) >= np.cos(np.deg2rad(theta))
    else:
        vx, vy = (nx, ny)
        porte = True
    yy, xx = np.mgrid[0:q.shape[0], 0:q.shape[1]].astype(np.float32)
    a = map_coordinates(qs, [yy + vy, xx + vx], order=1, mode='nearest')
    b = map_coordinates(qs, [yy - vy, xx - vx], order=1, mode='nearest')
    cand = (qs >= a) & (qs >= b) & (2 * qs - a - b > kc * qs) & porte & m
    if loc > 0:
        amp = gaussian_filter(ndi.maximum_filter(qs, 25), 12)
        qn = qs / np.maximum(amp, loc * np.percentile(qs[m], 99.5))
    else:
        qn = qs
    T = threshold_otsu(qn[m])
    faible = cand & (qn > lo * T)
    lab, n = ndi.label(faible, C8)
    if n == 0:
        return np.zeros(q.shape, np.uint8)
    fort = ndi.maximum(cand & (qn > hi * T), lab, np.arange(1, n + 1)) > 0
    S = np.isin(lab, np.nonzero(fort)[0] + 1)
    S = skeletonize(ndi.binary_closing(S, np.ones((2, 2))) & m)
    if ep > 0:
        S = elaguer(S, ep)
    S = couper_jonctions(S)
    if gap > 0:
        S = skeletonize(combler(S, m, gap, -ny, nx))
        S = couper_jonctions(elaguer(S, ep) if ep > 0 else S)
    if lmin > 0:
        lab, n = ndi.label(S, C8)
        if n:
            t = ndi.sum(np.ones(lab.shape, np.uint8), lab, np.arange(1, n + 1))
            S = np.isin(lab, np.nonzero(t >= lmin)[0] + 1)
    return (S & m).astype(np.uint8) * 255


def extract_lines(prob, mask=None, **kw):
    """prob (H, W) probability map (any range), mask (H, W) bool or None (= whole image) -> uint8 (H, W), 255 on the extracted lines"""
    m = np.ones(prob.shape, bool) if mask is None else np.asarray(mask, bool)
    return nms2(np.asarray(prob, np.float32), m, **{**SETTINGS, **kw})


def topology(lines):
    """counts on a 1-px line mask: pixels, junction pixels (>= 3 neighbours), line ends (1 neighbour), pieces"""
    S = np.asarray(lines) > 0; nb = voisins(S); lab, n = ndi.label(S, C8)
    return dict(pixels=int(S.sum()), junctions=int((S & (nb >= 3)).sum()), ends=int((S & (nb == 1)).sum()), pieces=int(n))

