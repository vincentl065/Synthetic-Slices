"""Extra phenomena (Synthetic-Slices v0.2), labels always exact:
  - bord: scroll edge, the last sheets fray out into air (layout changed BEFORE the diffusion model, air repainted after);
  - contact: neighbouring sheets pressed together (3-8 px);
  - peindre: very bright mineral inclusions and debris in the air, painted after (no line is changed).
Internal names are French (original research code)."""
import numpy as np
from scipy import ndimage as ndi

def bord(idm, rng):
    ks = np.unique(idm[idm > 0])
    if len(ks) < 4:
        return (idm, np.zeros(idm.shape, bool))
    sens = rng.random() < 0.5
    n_ret = int(rng.integers(max(1, len(ks) // 4), max(2, len(ks) // 2 + 1)))
    k0 = ks[-n_ret - 1] if sens else ks[n_ret]
    out = idm.copy()
    retire = idm > k0 if sens else (idm < k0) & (idm > 0)
    out[retire] = 0
    garde = out > 0
    d_ret = ndi.distance_transform_edt(~retire) if retire.any() else np.full(idm.shape, 1000000000.0)
    d_gar = ndi.distance_transform_edt(~garde)
    air = (d_ret < d_gar) & (d_gar > 5)
    dern = [k0] + ([k0 - 1] if sens else [k0 + 1]) * int(rng.random() < 0.6)
    for k in dern:
        ys, xs = np.nonzero(out == k)
        if len(ys) < 40:
            continue
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(len(ys)))
            r = rng.uniform(3, 20)
            m = np.hypot(np.arange(idm.shape[0])[:, None] - ys[j], np.arange(idm.shape[1])[None, :] - xs[j]) < r
            out[m & (out == k)] = 0
    lab, n = ndi.label(out > 0, np.ones((3, 3)))
    if n:
        tailles = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        petits = np.isin(lab, np.nonzero(tailles < 6)[0] + 1)
        out[petits] = 0
    return (out, air)

def contact(idm, rng, dmin=3.0):
    out = idm.copy()
    ks = [k for k in np.unique(idm) if k > 0]
    H, W = idm.shape
    nxt = int(idm.max()) + 1
    n_aj = 0
    if not ks:
        return (out, 0)
    for k in rng.choice(ks, max(1, int(len(ks) * rng.uniform(0.3, 0.6))), replace=False):
        sk = out == k
        if sk.sum() < 40:
            continue
        d = rng.uniform(3.0, 8.0)
        dk = ndi.distance_transform_edt(~sk)
        dau = ndi.distance_transform_edt(~((out > 0) & ~sk))
        bande = (np.abs(dk - d) < 0.7) & (dau > dmin + 0.5)
        lab, n = ndi.label(bande, np.ones((3, 3)))
        if n == 0:
            continue
        tailles = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        c = int(rng.choice(np.argsort(tailles)[-2:])) + 1
        m = lab == c
        if m.sum() < 30:
            continue
        ys, xs = np.nonzero(m)
        j = int(rng.integers(len(ys)))
        L = rng.uniform(30, 300)
        m &= np.hypot(np.arange(H)[:, None] - ys[j], np.arange(W)[None, :] - xs[j]) < L
        from skimage.morphology import skeletonize
        s = skeletonize(m)
        lab2, n2 = ndi.label(s, np.ones((3, 3)))
        if n2 == 0:
            continue
        t2 = ndi.sum(np.ones_like(lab2), lab2, np.arange(1, n2 + 1))
        s = lab2 == int(np.argmax(t2)) + 1
        if s.sum() < 20 or (ndi.distance_transform_edt(~(out > 0))[s] < dmin - 0.01).any():
            continue
        out[s] = nxt
        nxt += 1
        n_aj += 1
    return (out, n_aj)

def peindre(img, traits, air, rng, inclusions=True, debris=True):
    im = img.astype(np.float32)
    H, W = im.shape
    info = {}
    a_gris = float(np.percentile(im, 3))
    if air is not None and air.any():
        al = np.clip(ndi.gaussian_filter(air.astype(np.float32), 1.5) * 1.4, 0, 1)
        grain = a_gris + ndi.gaussian_filter(rng.normal(0, 6, im.shape), 0.8) * 2.5
        im = im * (1 - al) + grain * al
    loin = ndi.distance_transform_edt(~traits) > 5
    if debris:
        vide = loin & (im < np.percentile(im, 35))
        n = 0
        for _ in range(int(rng.integers(1, 5))):
            ys, xs = np.nonzero(vide)
            if not len(ys):
                break
            j = int(rng.integers(len(ys)))
            r = rng.uniform(3, 15)
            yy, xx = np.ogrid[:H, :W]
            f = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 2.5)
            f /= f.std() + 1e-09
            m = (np.hypot(yy - ys[j], xx - xs[j]) < r * (1 + 0.45 * f)) & loin
            g = np.percentile(im, rng.uniform(40, 65)) + ndi.gaussian_filter(rng.normal(0, 8, (H, W)), 0.7) * 2
            al = np.clip(ndi.gaussian_filter(m.astype(np.float32), 1.0) * 1.3, 0, 1)
            im = im * (1 - al) + g * al
            n += 1
        info['debris'] = n
    if inclusions:
        n = int(rng.integers(1, 7))
        hi = float(np.percentile(im, 99.9))
        yy, xx = np.mgrid[:H, :W]
        for _ in range(n):
            y, x = (rng.uniform(4, H - 4), rng.uniform(4, W - 4))
            s = rng.uniform(0.8, 2.5)
            amp = rng.uniform(1.15, 1.6) * hi - float(im[int(y), int(x)])
            im += max(amp, 20) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
        info['inclusions'] = n
    return (np.clip(im, 0, 255).astype(np.uint8), info)
