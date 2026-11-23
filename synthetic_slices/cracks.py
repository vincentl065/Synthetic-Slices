"""Realistic cracks and crushed regions (Synthetic-Slices v0.2).
  - volume_fissure2: a jagged break through the stack (notches, fault turned 4-22 deg, variable width 0-12 px); sheets are cut cleanly and
    offset across it; the labels stay exact.
  - peindre_fente: paints the crack AFTER the diffusion model, with the tile's own air grey, sharp edges and sometimes debris.
  - peindre_ecrase: "melted" crushed look (blur across sheets, local contrast x 0.55-0.8, voids filled, scanner grain), whole tile or a patch.
The lines are never moved by the painting steps. Internal names are French (original research code)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import find_contours
from . import layout as S4
from . import folds as SP
from . import hard_cases as SV

def _crans(rng, n):
    v = np.zeros(n)
    i = 0
    while i < n:
        i += int(rng.integers(10, 40))
        v[i:] += rng.choice([-1, 1]) * rng.uniform(1, 4)
    return v

def volume_fissure2(R, rng, N=(256, 256), marge=48, essais=60):
    H, W = (N, N) if np.isscalar(N) else (int(N[0]), int(N[1]))
    Hs, Ws = (H + 2 * marge, W + 2 * marge)
    Yc, Xc = np.mgrid[0:Hs, 0:Ws].astype(np.float64)
    for _ in range(essais):
        xr, yr = SV._plan(R, rng, Hs, Ws)
        compact = rng.random() < 0.4
        f = yr
        lv = SP._niveaux(R, rng, float(f.min()), float(f.max()), None, compact, 1.0)
        f = SP._desordre_f(rng, f, lv, compact)
        gap = float(np.median(np.diff(lv)))
        gy, gx = np.gradient(f)
        cy, cx = (marge + rng.uniform(0.25, 0.75) * H, marge + rng.uniform(0.25, 0.75) * W)
        phi = np.arctan2(gy[int(cy), int(cx)], gx[int(cy), int(cx)]) + np.radians(rng.uniform(-35, 35))
        a = (Xc - cx) * np.cos(phi) + (Yc - cy) * np.sin(phi)
        b = -(Xc - cx) * np.sin(phi) + (Yc - cy) * np.cos(phi)
        La = int(np.ceil(np.abs(a).max())) + 4
        ax_ = np.arange(-La, La + 1)
        n_ = 2 * La + 1
        prof = rng.uniform(2, 10) * SV._bruit1(rng, n_, rng.uniform(30, 80)) + rng.uniform(1, 4) * SV._bruit1(rng, n_, rng.uniform(2, 5)) + _crans(rng, n_)
        bb = b - np.interp(a, ax_, prof)
        al = np.radians(rng.choice([-1, 1]) * rng.uniform(4, 22))
        ca, sa = (np.cos(al), np.sin(al))
        Xr = cx + (Xc - cx) * ca - (Yc - cy) * sa
        Yr = cy + (Xc - cx) * sa + (Yc - cy) * ca
        fr = map_coordinates(f, [Yr, Xr], order=1, mode='nearest')
        D0 = rng.choice([-1, 1]) * rng.uniform(0.3, 2.2) * gap
        Lt = max(40.0, 2.5 * abs(D0))
        a0, a1 = (-rng.uniform(80, 600), rng.uniform(80, 600))
        t = np.clip(np.minimum(a - a0, a1 - a) / Lt, 0, 1)
        f = np.where(bb > 0, f + t * (fr - f) + D0 * t, f)
        w0 = rng.choice([rng.uniform(0, 2), rng.uniform(2, 12)], p=[0.3, 0.7])
        mod = np.clip(0.55 + 0.6 * np.interp(a, ax_, SV._bruit1(rng, n_, rng.uniform(6, 15))), 0, 1.7)
        w = w0 * t ** 0.6 * mod
        eg = 0.8 * np.interp(a, ax_, SV._bruit1(rng, n_, 1.5))
        ed = 0.8 * np.interp(a, ax_, SV._bruit1(rng, n_, 1.5))
        bord = np.where(bb > 0, w / 2 + eg, w / 2 + ed)
        fente = (t > 0) & (np.abs(bb) <= bord) & (w > 0.6)
        seuil = np.where(t > 0, np.maximum(bord, 1.5), 0.0)
        cote = np.sign(bb)
        dist = np.abs(bb)
        bouts_ok = (t > 0) & (dist <= np.maximum(bord, 1.5) + 5 + 4)
        info = dict(genre='fissure2', compact=compact, gap=gap, decalage_ecarts=float(D0 / gap), rotation_deg=float(np.degrees(al)), largeur_max=float(w0))
        lignes, axe_id, couche = ([], np.zeros((H, W), np.uint16), np.zeros((H, W), np.uint8))

        def poser(k, P):
            for Q in S4._decouper(P, W, H):
                if len(Q) < 3:
                    continue
                lignes.append((k, Q))
                px_ = S4._pixels(Q, H, W)
                axe_id[px_[:, 0], px_[:, 1]] = k
                couche[px_[:, 0], px_[:, 1]] = 1
        k = 0
        recul = {}
        for L_ in lv[1:-1]:
            k += 1
            for C in find_contours(f, L_):
                s_ = map_coordinates(seuil, [C[:, 0], C[:, 1]], order=1)
                cs = np.sign(map_coordinates(cote, [C[:, 0], C[:, 1]], order=0))
                r = np.array([recul.setdefault((k, int(v)), rng.uniform(0, 5)) for v in cs])
                dd = map_coordinates(dist, [C[:, 0], C[:, 1]], order=1)
                keep = (s_ <= 0) | (dd > s_ + np.where(s_ > 0, r, 0))
                for seg in np.split(np.nonzero(keep)[0], np.nonzero(np.diff(np.nonzero(keep)[0]) > 1)[0] + 1):
                    if len(seg) > 3:
                        poser(k, np.stack([C[seg, 1] - marge, C[seg, 0] - marge], 1))
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
        if c['croisements'] or c['fourches'] or c['boucles'] or mauvais or (c['ecart_p10'] < S4.GMIN - 2.0) or (c['distance_min_entre_feuilles'] < 3.0):
            continue
        if c['bouts_interieurs'] == 0:
            continue
        info['bouts_interieurs'] = c['bouts_interieurs']
        return dict(fente=fente[marge:marge + H, marge:marge + W].astype(np.uint8)[None], axe_id=axe_id[None], couche=couche[None], axe=(axe_id > 0).astype(np.uint8)[None], lignes=[lignes], genre='fissure2', controle=c, n_pile=len(us), compact=compact, info=info)
    return None

def _air(im, traits):
    loin = ndi.distance_transform_edt(~traits) > 4
    v = im[loin & (im < np.percentile(im, 25))] if traits.any() else im[im < np.percentile(im, 10)]
    if v.size < 200:
        v = im[im < np.percentile(im, 8)]
    return (float(np.median(v)), float(max(3.0, min(14.0, v.std()))))

def peindre_fente(img, fente, traits, rng):
    im = img.astype(np.float32)
    H, W = im.shape
    if not fente.any():
        return (img, {})
    a0, sd = _air(im, traits)
    grain = a0 + gaussian_filter(rng.normal(0, 1, im.shape), 0.7) * sd * 1.6
    al = np.clip(gaussian_filter(fente.astype(np.float32), 0.5) * 1.3, 0, 1)
    im = im * (1 - al) + grain * al
    info = dict(debris=0)
    if rng.random() < 0.45:
        f2 = gaussian_filter(rng.normal(0, 1, im.shape), rng.uniform(2, 5))
        f2 /= f2.std() + 1e-09
        m = fente & (f2 > np.percentile(f2[fente], rng.uniform(30, 80)))
        m = ndi.binary_opening(m, iterations=1)
        g = np.percentile(img, rng.uniform(30, 60)) + gaussian_filter(rng.normal(0, 1, im.shape), 0.8) * sd * 2.5
        a2 = np.clip(gaussian_filter(m.astype(np.float32), 0.6) * 1.3, 0, 1)
        im = im * (1 - a2) + g * a2
        ys, xs = np.nonzero(fente)
        for _ in range(int(rng.integers(0, 4))):
            j = int(rng.integers(len(ys)))
            yy, xx = np.ogrid[:H, :W]
            s = rng.uniform(0.7, 1.6)
            im += rng.uniform(40, 90) * np.exp(-((yy - ys[j]) ** 2 + (xx - xs[j]) ** 2) / (2 * s * s))
        info['debris'] = int(m.sum())
    return (np.clip(im, 0, 255).astype(np.uint8), info)

def _normale(traits):
    d = np.minimum(ndi.distance_transform_edt(~traits), 12.0)
    gy, gx = np.gradient(gaussian_filter(d, 1.0))
    J = [gaussian_filter(q, 4) for q in (gx * gx, gy * gy, gx * gy)]
    phi = 0.5 * np.arctan2(2 * J[2], J[0] - J[1])
    return (np.cos(phi), np.sin(phi))

def peindre_ecrase(img, traits, rng, plaque=False):
    im = img.astype(np.float32)
    H, W = im.shape
    if traits.sum() < 50:
        return (img, {})
    c, s = _normale(traits)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sig = rng.uniform(0.5, 1.0)
    acc = np.zeros_like(im)
    ws = 0.0
    for o in np.arange(-2.5, 2.51, 0.5):
        w = np.exp(-0.5 * (o / sig) ** 2)
        acc += w * map_coordinates(im, [yy + o * s, xx + o * c], order=1, mode='reflect')
        ws += w
    fl = acc / ws
    m = gaussian_filter(fl, 5)
    k = rng.uniform(0.55, 0.8)
    out = m + (fl - m) * k
    out = np.maximum(out, m - rng.uniform(14, 24))
    a0, sd = _air(im, traits)
    st = gaussian_filter(rng.normal(0, 1, im.shape), 0.6).astype(np.float32)
    acc = np.zeros_like(st)
    for o in np.arange(-2, 2.01, 1.0):
        acc += map_coordinates(st, [yy + o * c, xx - o * s], order=1, mode='reflect')
    out = out + acc / 5.0 * sd * rng.uniform(0.8, 1.4)
    out = out + gaussian_filter(rng.normal(0, 1, im.shape), 0.5) * sd * rng.uniform(0.6, 1.0)
    if plaque:
        f = gaussian_filter(rng.normal(0, 1, (H, W)), rng.uniform(25, 45))
        f /= f.std() + 1e-09
        z = f > np.percentile(f, rng.uniform(10, 60))
        al = np.clip(gaussian_filter(z.astype(np.float32), 4), 0, 1)
    else:
        al = np.ones_like(im)
    res = im * (1 - al) + out * al
    return (np.clip(res, 0, 255).astype(np.uint8), dict(flou=float(sig), contraste=float(k), part=float(al.mean())))
