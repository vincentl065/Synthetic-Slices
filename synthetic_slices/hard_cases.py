"""Hard-case layouts (Synthetic-Slices v0.2): "fissure" (first version of the crack), "mush" (crushed sheets with a slow,
weak, patchy waviness) and "delamination" (layers splitting apart), all as continuous sheet lines with exact ids.
Public entry point: volume_v2(settings, rng, size, genre, margin) with genre in GENRES_V2.
Internal names are French (original research code)."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import find_contours
from . import layout as S4
from . import folds as SP
GENRES_V2 = ['fissure', 'mush', 'delamination']

def _bruit1(rng, n, sig):
    v = gaussian_filter(rng.normal(0, 1, n), sig)
    return v / (v.std() + 1e-09)

def _plan(R, rng, Hs, Ws):
    Xd, Yd = SP._deformation(rng, Hs, Ws)
    R2 = dict(R)
    R2['virage'] = R['virage'] * 0.5
    R2['n_virages'] = 1
    ops = S4._virages(R2, rng, 1, 1.5 * np.hypot(Hs, Ws) + 400)
    Xu, Yu = S4._deplier(Xd.ravel(), Yd.ravel(), 0.0, ops, 0)
    Xu, Yu = (Xu.reshape(Xd.shape), Yu.reshape(Yd.shape))
    th = rng.uniform(0, 2 * np.pi)
    c, s = (np.cos(th), np.sin(th))
    return (Xu * c + Yu * s, -Xu * s + Yu * c)

def _mush(rng, xr, yr, Hs, Ws):
    P = rng.uniform(30, 70)
    A0 = rng.uniform(1.5, 3.5)
    L1 = int(np.ceil(np.abs(xr).max())) + 4
    n1 = lambda sig: np.interp(xr, np.arange(-L1, L1 + 1), _bruit1(rng, 2 * L1 + 1, sig))
    Am = A0 * np.clip(0.6 + 0.6 * SP._champ_lisse(rng, (Hs, Ws), rng.uniform(80, 160)), 0.0, 1.6)
    t = xr / P + 0.3 * SP._champ_lisse(rng, (Hs, Ws), rng.uniform(150, 250)) + rng.uniform(0.15, 0.35) * n1(rng.uniform(40, 80))
    return yr + Am * np.sin(2 * np.pi * t)

def volume_v2(R, rng, N=(512, 512), genre='fissure', marge=48, essais=60):
    H, W = (N, N) if np.isscalar(N) else (int(N[0]), int(N[1]))
    Hs, Ws = (H + 2 * marge, W + 2 * marge)
    Yc, Xc = np.mgrid[0:Hs, 0:Ws].astype(np.float64)
    for _ in range(essais):
        xr, yr = _plan(R, rng, Hs, Ws)
        compact = genre == 'mush' or (genre == 'fissure' and rng.random() < 0.4)
        f = _mush(rng, xr, yr, Hs, Ws) if genre == 'mush' else yr
        ech = 1.0
        if genre == 'mush':
            gy_, gx_ = np.gradient(f)
            ech = 0.85 * float(np.percentile(np.hypot(gy_, gx_), 90))
        lv = SP._niveaux(R, rng, float(f.min()), float(f.max()), None, compact, ech)
        f = SP._desordre_f(rng, f, lv, compact)
        gap = float(np.median(np.diff(lv)))
        info = dict(genre=genre, compact=compact, gap=gap)
        seuil = np.zeros((Hs, Ws))
        cote = np.zeros((Hs, Ws))
        bouts_ok = np.zeros((Hs, Ws), bool)
        arcs = []
        fente = np.zeros((Hs, Ws), bool)
        if genre == 'fissure':
            gy, gx = np.gradient(f)
            cy, cx = (marge + rng.uniform(0.2, 0.8) * H, marge + rng.uniform(0.2, 0.8) * W)
            phi = np.arctan2(gy[int(cy), int(cx)], gx[int(cy), int(cx)]) + np.radians(rng.uniform(-35, 35))
            a = (Xc - cx) * np.cos(phi) + (Yc - cy) * np.sin(phi)
            b = -(Xc - cx) * np.sin(phi) + (Yc - cy) * np.cos(phi)
            La = int(np.ceil(np.abs(a).max())) + 4
            ax_ = np.arange(-La, La + 1)
            bc = np.interp(a, ax_, rng.uniform(2, 10) * _bruit1(rng, 2 * La + 1, rng.uniform(30, 80)))
            bb = b - bc
            D0 = rng.choice([-1, 1]) * rng.uniform(0.3, 2.2) * gap
            Lt = max(40.0, 2.5 * abs(D0))
            a0, a1 = (-rng.uniform(80, 600), rng.uniform(80, 600))
            t = np.clip(np.minimum(a - a0, a1 - a) / Lt, 0, 1)
            w0 = rng.uniform(3, 16)
            w = w0 * t ** 0.6 * np.clip(1 + 0.35 * np.interp(a, ax_, _bruit1(rng, 2 * La + 1, 25)), 0.4, 1.6)
            f = f + D0 * t * (bb > 0)
            cote = np.sign(bb)
            seuil = np.where(t > 0, np.maximum(w / 2, 1.5), 0.0)
            dist = np.abs(bb)
            bouts_ok = dist <= w / 2 + 7 + 4
            fente = (t > 0) & (dist <= w / 2)
            info.update(decalage_ecarts=float(D0 / gap), largeur_max=float(w0), angle_deg=float(np.degrees(phi)), pointes=(float(a0), float(a1)))
        if genre == 'delamination':
            i = int(rng.integers(max(1, len(lv) // 3), max(2, 2 * len(lv) // 3)))
            Li = float(lv[i])
            pts = np.argwhere((np.abs(f - Li) < 0.6) & (Yc > marge + 0.15 * H) & (Yc < marge + 0.85 * H) & (Xc > marge + 0.15 * W) & (Xc < marge + 0.85 * W))
            if not len(pts):
                continue
            py, px = pts[rng.integers(len(pts))]
            gy, gx = np.gradient(f)
            tg = np.arctan2(gy[py, px], gx[py, px]) + np.pi / 2
            du, dv = (Xc - px, Yc - py)
            au = du * np.cos(tg) + dv * np.sin(tg)
            bv = -du * np.sin(tg) + dv * np.cos(tg)
            sl, sa = (rng.uniform(40, 90), rng.uniform(18, 32))
            g = np.exp(-0.5 * ((au / sl) ** 2 + (bv / sa) ** 2))
            wl = rng.uniform(0.6, 0.9) * gap
            h = rng.uniform(0.72, 0.85) * wl
            s_ = f - Li
            psi = np.tanh(np.maximum(s_, 0) / wl) * np.exp(-(np.maximum(s_ - 2 * wl, 0) / (3 * gap)) ** 2)
            f = f - h * g * psi
            info.update(feuille_niveau=i, longueur=float(2.5 * sl), ouverture=float(h), largeur=float(wl))
            for C in find_contours(f, Li + 1.6):
                Pm = map_coordinates(f, [C[:, 0], C[:, 1]], order=1)
                d_main = np.abs(map_coordinates(f - Li, [C[:, 0], C[:, 1]], order=1)) / np.maximum(map_coordinates(np.hypot(*np.gradient(f)), [C[:, 0], C[:, 1]], order=1), 0.001)
                k_ = (d_main >= 3.5) & (map_coordinates(g, [C[:, 0], C[:, 1]], order=1) > 0.05)
                for seg in np.split(np.nonzero(k_)[0], np.nonzero(np.diff(np.nonzero(k_)[0]) > 1)[0] + 1):
                    if len(seg) > 12:
                        arcs.append(C[seg])
            if not arcs:
                continue
            bouts_ok = g > 0.03
        lignes, axe_id, couche = ([], np.zeros((H, W), np.uint16), np.zeros((H, W), np.uint8))

        def poser(k, P, c=1):
            for Q in S4._decouper(P, W, H):
                if len(Q) < 3:
                    continue
                lignes.append((k, Q))
                px_ = S4._pixels(Q, H, W)
                axe_id[px_[:, 0], px_[:, 1]] = k
                couche[px_[:, 0], px_[:, 1]] = c
        k = 0
        recul = {}
        for L_ in lv[1:-1]:
            k += 1
            for C in find_contours(f, L_):
                if genre == 'fissure':
                    s_ = map_coordinates(seuil, [C[:, 0], C[:, 1]], order=1)
                    cs = np.sign(map_coordinates(cote, [C[:, 0], C[:, 1]], order=0))
                    r = np.array([recul.setdefault((k, int(v)), rng.uniform(0, 7)) for v in cs])
                    dd = np.abs(map_coordinates(bb, [C[:, 0], C[:, 1]], order=1))
                    keep = (s_ <= 0) | (dd > s_ + np.where(s_ > 0, r, 0))
                    for seg in np.split(np.nonzero(keep)[0], np.nonzero(np.diff(np.nonzero(keep)[0]) > 1)[0] + 1):
                        if len(seg) > 3:
                            poser(k, np.stack([C[seg, 1] - marge, C[seg, 0] - marge], 1))
                else:
                    poser(k, np.stack([C[:, 1] - marge, C[:, 0] - marge], 1))
            if genre == 'delamination' and abs(L_ - Li) < 1e-06:
                for C in arcs:
                    poser(k, np.stack([C[:, 1] - marge, C[:, 0] - marge], 1), 2)
        if not lignes:
            continue
        us = sorted({kk for kk, _ in lignes})
        num = {kk: j + 1 for j, kk in enumerate(us)}
        lignes = [(num[kk], Q) for kk, Q in lignes]
        lut = np.zeros(int(axe_id.max()) + 1, np.uint16)
        for kk, v in num.items():
            lut[kk] = v
        axe_id = lut[axe_id]
        c = S4.controle(axe_id, lignes)
        bo = bouts_ok[marge:marge + H, marge:marge + W]
        mauvais = 0
        for _, Q in lignes:
            for p in (Q[0], Q[-1]):
                if min(p[0] + 0.5, W - 0.5 - p[0], p[1] + 0.5, H - 0.5 - p[1]) <= 2.0:
                    continue
                if not bo[int(np.clip(round(p[1]), 0, H - 1)), int(np.clip(round(p[0]), 0, W - 1))]:
                    mauvais += 1
        gmin = S4.GMIN - (2.5 if genre == 'mush' else 2.0)
        if c['croisements'] or c['fourches'] or c['boucles'] or mauvais or (c['ecart_p10'] < gmin) or (c['distance_min_entre_feuilles'] < 3.0):
            continue
        if genre == 'fissure' and c['bouts_interieurs'] == 0:
            continue
        info['bouts_interieurs'] = c['bouts_interieurs']
        return dict(fente=fente[marge:marge + H, marge:marge + W].astype(np.uint8)[None], axe_id=axe_id[None], couche=couche[None], axe=(axe_id > 0).astype(np.uint8)[None], lignes=[lignes], genre=genre, controle=c, n_pile=len(us), compact=compact, info=info)
    return None
