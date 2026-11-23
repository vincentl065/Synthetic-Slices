"""Fold layouts (Synthetic-Slices v0.2): hairpins, tight U folds, tight bends and accordion folds, drawn as continuous
sheet lines that never fork or cross. A stack from layout.py is bent by a smooth, invertible deformation; sheets are grouped into packets at the
large gaps, and an optional "compact" mode packs every sheet close to the minimum gap.
Public entry point: volume_pli(settings, rng, size, genre, margin, compact=False) with genre in GENRES.
Internal names are French (original research code)."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import find_contours
from . import layout as S4
GENRES = ['epingle', 'U_serre', 'virage_serre', 'accordeon']

def _niveaux(R, rng, fmin, fmax, r0=None, compact=False, ech=1.0):
    Y = [fmin if r0 is None else r0]
    if compact:
        while Y[-1] < fmax:
            n = max(3, int(round(rng.gamma(2.0, R['paquet'] / 2))))
            for _ in range(n - 1):
                Y.append(Y[-1] + ech * (S4.GMIN + 0.4 + rng.exponential(1.2)))
            Y.append(Y[-1] + ech * (S4.GMIN + 1.0 + rng.exponential(2.5)))
        return np.array(Y)
    while Y[-1] < fmax:
        n = max(2, int(round(rng.gamma(2.0, R['paquet'] / 2))))
        b = R['gap'] * np.exp(rng.normal(0, 0.12))
        ecrase = rng.random() < 0.25
        for _ in range(n - 1):
            g = S4.GMIN + 0.4 + rng.exponential(1.5) if ecrase else max(S4.GMIN + 0.4, b * np.exp(rng.normal(0, R['gap_var'] + 0.1)))
            Y.append(Y[-1] + g)
        Y.append(Y[-1] + b * R['inter'] * np.exp(rng.normal(0, 0.35)) * (rng.uniform(1.3, 2.5) if rng.random() < 0.1 else 1.0))
    return np.array(Y)

def _champ_lisse(rng, shape, sigma, pas=8):
    h, w = (shape[0] // pas + 2, shape[1] // pas + 2)
    n = gaussian_filter(rng.normal(0, 1, (h, w)), sigma / pas)
    n /= n.std() + 1e-09
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]] / pas
    return map_coordinates(n, [yy, xx], order=1)

def _deformation(rng, Hs, Ws, fort=False):
    Y, X = np.mgrid[0:Hs, 0:Ws].astype(np.float64)
    X -= Ws / 2
    Y -= Hs / 2
    a = rng.uniform(0, np.pi)
    s = rng.uniform(0.4, 0.75) if fort else rng.uniform(0.65, 1.0)
    c, sn = (np.cos(a), np.sin(a))
    u, v = (X * c + Y * sn, -X * sn + Y * c)
    u = u / s
    X, Y = (u * c - v * sn, u * sn + v * c)
    sig = rng.uniform(70, 130)
    A = rng.uniform(25, 50) if fort else rng.uniform(8, 28)
    dx, dy = (_champ_lisse(rng, (Hs, Ws), sig), _champ_lisse(rng, (Hs, Ws), sig))
    pente = max(np.abs(np.gradient(dx)).max(), np.abs(np.gradient(dy)).max()) * A
    A = A * min(1.0, 0.45 / max(pente, 1e-09))
    X, Y = (X + A * dx, Y + A * dy)
    return (X, Y)

def _desordre_f(rng, f, lv, compact=False):
    gap = float(np.median(np.diff(lv)))
    Hs, Ws = f.shape
    nf = int((f.max() - f.min()) / gap) + 4
    pas = 8
    h, w = (Hs // pas + 2, Ws // pas + 2)
    sxy = rng.uniform(40, 90) / pas
    sfb = rng.uniform(1.2, 2.0)
    N = gaussian_filter(rng.normal(0, 1, (nf, h, w)), (sfb, sxy, sxy))
    N /= N.std() + 1e-09
    amp = rng.uniform(0.25, 0.45) * gap
    pente_f = np.abs(np.diff(N, axis=0)).max() * amp / gap if nf > 1 else 0
    amp *= min(1.0, 0.6 / max(pente_f, 1e-09))
    yy, xx = np.mgrid[0:Hs, 0:Ws] / pas
    ff = (f - f.min()) / gap + 1
    f2 = f + amp * map_coordinates(N, [ff, yy, xx], order=1, mode='nearest')
    gy, gx = np.gradient(f)
    for _ in range(0 if compact else rng.poisson(0.8)):
        i = int(rng.integers(1, len(lv) - 1))
        fl = 0.5 * (lv[i] + lv[i - 1])
        pts = np.argwhere(np.abs(f - fl) < gap * 0.3)
        if not len(pts):
            continue
        cy, cx = pts[rng.integers(len(pts))]
        t = np.arctan2(gy[cy, cx], gx[cy, cx]) + np.pi / 2
        sl, sa = (rng.uniform(60, 160), rng.uniform(25, 55))
        Yc, Xc = np.mgrid[0:Hs, 0:Ws]
        du, dv = (Xc - cx, Yc - cy)
        a_ = du * np.cos(t) + dv * np.sin(t)
        b_ = -du * np.sin(t) + dv * np.cos(t)
        g = np.exp(-0.5 * ((a_ / sl) ** 2 + (b_ / sa) ** 2))
        w = rng.uniform(0.6, 1.6) * gap
        d = rng.uniform(0.5, 0.85) * w
        f2 = f2 - d * g * np.tanh((f - fl) / w)
    return _paquets(rng, f2, lv, compact)

def _paquets(rng, f, lv, compact=False):
    d = np.diff(lv)
    med = float(np.median(d))
    Hs, Ws = f.shape
    bords = [lv[0] - med] + [0.5 * (lv[i] + lv[i + 1]) for i in range(len(d)) if d[i] > (1.15 if compact else 1.25) * med] + [lv[-1] + med]
    out = f.copy()
    for b1, b2 in zip(bords[:-1], bords[1:]):
        m = (f > b1) & (f < b2)
        if not m.any():
            continue
        gam = np.exp(rng.uniform(0.3, 0.8) * _champ_lisse(rng, (Hs, Ws), rng.uniform(80, 160)) + rng.normal(0, 0.2))
        gam = np.clip(gam, 0.7, 1.4) if compact else np.clip(gam, 0.45, 2.2)
        g = (f[m] - b1) / (b2 - b1)
        out[m] = b1 + (b2 - b1) * g ** gam[m]
    return out

def _champ(genre, R, rng, X, Y, Hs, Ws):
    th = rng.uniform(0, 2 * np.pi)
    c, s = (np.cos(th), np.sin(th))
    cx, cy = (rng.uniform(-0.25, 0.25) * Ws, rng.uniform(-0.25, 0.25) * Hs)
    x, y = (X - cx, Y - cy)
    xr, yr = (x * c + y * s, -x * s + y * c)
    if genre in ('epingle', 'U_serre'):
        if genre == 'epingle':
            r0 = rng.uniform(10, 24)
            ouv = rng.uniform(0.0, 0.35)
        else:
            r0 = rng.uniform(5, 10)
            ouv = rng.uniform(0.0, 0.12)
        sp = 40.0 * np.logaddexp(0, -xr / 40.0)
        yb = yr / (1 + ouv * sp / 200.0)
        dis = rng.uniform(-0.25, 0.25)
        k = np.where(yr > 0, 1 + dis, 1 - dis)
        f = np.where(xr < 0, np.abs(yb) * k, np.hypot(xr, yb * k))
        return (f, r0)
    if genre == 'accordeon':
        P = rng.uniform(40, 80)
        A0 = rng.uniform(10, 20)

        def n1(sig):
            L1 = int(np.ceil(np.abs(xr).max())) + 4
            v = gaussian_filter(rng.normal(0, 1, 2 * L1 + 1), sig)
            v /= v.std() + 1e-09
            return np.interp(xr, np.arange(-L1, L1 + 1), v)
        Am = A0 * np.clip(1 + 0.3 * _champ_lisse(rng, (Hs, Ws), rng.uniform(160, 240)), 0.5, 1.5) * np.clip(1 + 0.55 * n1(rng.uniform(60, 110)), 0.15, 1.7)
        ph = 0.2 * _champ_lisse(rng, (Hs, Ws), rng.uniform(200, 280)) + rng.uniform(0.3, 0.6) * n1(rng.uniform(50, 90))
        t = xr / P + ph
        k = np.clip(0.75 + 0.2 * n1(rng.uniform(80, 140)), 0.4, 0.97)
        z = np.arcsin(k * np.sin(2 * np.pi * t)) / np.arcsin(k)
        f = yr + Am * z
        return (f, None)
    A = rng.uniform(0.35, 0.7) * np.pi
    Rs = rng.uniform(0.35, 0.6) * min(Hs, Ws)
    r = np.hypot(x, y)
    a = -A * np.exp(-(r / Rs) ** 2) * rng.choice([-1, 1])
    ca, sa = (np.cos(a), np.sin(a))
    xu, yu = (x * ca - y * sa, x * sa + y * ca)
    return (-xu * s + yu * c, None)

def _inverse(Xd, Yd, P, Hs, Ws):
    Q = P.copy()
    for _ in range(30):
        ix = np.clip(Q[:, 0] + Ws / 2, 0, Ws - 1.001)
        iy = np.clip(Q[:, 1] + Hs / 2, 0, Hs - 1.001)
        fx = map_coordinates(Xd, [iy, ix], order=1)
        fy = map_coordinates(Yd, [iy, ix], order=1)
        Q[:, 0] += 0.8 * (P[:, 0] - fx)
        Q[:, 1] += 0.8 * (P[:, 1] - fy)
    ok = (np.abs(P[:, 0] - fx) < 1.0) & (np.abs(P[:, 1] - fy) < 1.0)
    return (Q, ok)

def volume_pli(R, rng, N=(512, 512), genre='epingle', marge=48, essais=40, cisaillement=0.5, compact=False):
    H, W = (N, N) if np.isscalar(N) else (int(N[0]), int(N[1]))
    Hs, Ws = (H + 2 * marge, W + 2 * marge)
    for _ in range(essais):
        Xd, Yd = _deformation(rng, Hs, Ws, fort=genre == 'tourbillon')
        R2 = dict(R)
        R2['virage'] = R['virage'] * cisaillement
        R2['n_virages'] = 1
        ops = S4._virages(R2, rng, 1, 1.5 * np.hypot(Hs, Ws) + 400)
        Xu, Yu = S4._deplier(Xd.ravel(), Yd.ravel(), 0.0, ops, 0)
        Xu, Yu = (Xu.reshape(Xd.shape), Yu.reshape(Yd.shape))
        lignes, k = ([], 0)
        axe_id = np.zeros((H, W), np.uint16)
        ok = True

        def poser(k, P):
            for Q in S4._decouper(P, W, H):
                if len(Q) < 3:
                    continue
                lignes.append((k, Q))
                px = S4._pixels(Q, H, W)
                axe_id[px[:, 0], px[:, 1]] = k
        if genre == 'tourbillon':
            nb = int(rng.integers(3, 8))
            g1 = _niveaux(R, rng, 0.0, nb * R['gap'] * 1.6)[:nb + 1]
            per = float(g1[-1] - g1[0])
            bpas = per / (2 * np.pi)
            sens = rng.choice([-1, 1])
            ph = rng.uniform(0, 2 * np.pi)
            cx, cy = (rng.uniform(-0.45, 0.45) * Ws, rng.uniform(-0.45, 0.45) * Hs)
            r0 = max(12.0, 1.5 * bpas)
            rmax = 0.8 * np.hypot(Hs, Ws)
            tmax = (rmax - r0) / bpas + 2 * np.pi
            th = np.arange(0, tmax, 0.6 / rmax * 4)
            nq = int(tmax / (2 * np.pi) * nb) + nb + 4
            nphi = 64
            Nq = gaussian_filter(rng.normal(0, 1, (nq, nphi)), (rng.uniform(1.2, 2.0), rng.uniform(3, 7)), mode=('nearest', 'wrap'))
            Nq /= Nq.std() + 1e-09
            amp = rng.uniform(0.4, 0.6) * per / nb
            amp *= min(1.0, 0.6 / max(np.abs(np.diff(Nq, axis=0)).max() * amp / (per / nb), 1e-09))
            for i, d in enumerate(g1[:-1] - g1[0]):
                rb = r0 + bpas * th
                phi = (th + 2 * np.pi * d / per) % (2 * np.pi)
                q = th / (2 * np.pi) * nb + i
                nr = amp * map_coordinates(Nq, [q, phi / (2 * np.pi) * nphi], order=1, mode='grid-wrap')
                r = rb + nr * np.clip((rb - r0) / (1.5 * per), 0, 1)
                a_ = sens * (th + 2 * np.pi * d / per) + ph
                P = np.stack(S4._plier(cx + r * np.cos(a_), cy + r * np.sin(a_), 0.0, ops, 0), 1)
                Q, bon = _inverse(Xd, Yd, P, Hs, Ws)
                for seg in np.split(np.nonzero(bon)[0], np.nonzero(np.diff(np.nonzero(bon)[0]) > 1)[0] + 1):
                    if len(seg) > 2:
                        poser(i + 1, np.stack([Q[seg, 0] + Ws / 2 - marge, Q[seg, 1] + Hs / 2 - marge], 1))
        else:
            f, r0 = _champ(genre, R, rng, Xu, Yu, Hs, Ws)
            ech = 1.0
            if genre == 'accordeon':
                gy_, gx_ = np.gradient(f)
                ech = float(np.percentile(np.hypot(gy_, gx_), 90))
            lv = _niveaux(R, rng, float(f.min()), float(f.max()), r0, compact, ech)
            f = _desordre_f(rng, f, lv, compact)
            for L_ in lv[:-1] if r0 is not None else lv[1:-1]:
                k += 1
                for C in find_contours(f, L_):
                    poser(k, np.stack([C[:, 1] - marge, C[:, 0] - marge], 1))
        if not lignes:
            continue
        us = sorted({kk for kk, _ in lignes})
        num = {kk: i + 1 for i, kk in enumerate(us)}
        lignes = [(num[kk], Q) for kk, Q in lignes]
        lut = np.zeros(int(axe_id.max()) + 1, np.uint16)
        for kk, v in num.items():
            lut[kk] = v
        axe_id = lut[axe_id]
        c = S4.controle(axe_id, lignes)
        bouts_ok = c['bouts_interieurs'] == 0 or (genre == 'tourbillon' and c['bouts_interieurs'] == c['bouts_dans_un_vide'] and (c['bouts_interieurs'] <= len(us)))
        if c['croisements'] or c['fourches'] or c['boucles'] or (not bouts_ok) or (c['ecart_p10'] < S4.GMIN - 2) or (c['distance_min_entre_feuilles'] < 3.0):
            ok = False
        if ok:
            return dict(axe_id=axe_id[None], axe=(axe_id > 0).astype(np.uint8)[None], lignes=[lignes], genre=genre, controle=c, n_pile=len(us), compact=compact)
    return None
