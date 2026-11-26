"""Procedural sheet layouts (step 1 of Synthetic-Slices).

A flat stack of sheets is built in an "unfolded" (u, v) plane, then deformed and folded without crossings:
  - packets: sheets are grouped (gamma-distributed packet size), tight inside a packet, wider gaps between packets;
  - lens-shaped voids: a gap opens like an almond, neighbouring sheets bend around it and the bump decays over several sheets;
  - pinches / fans: a group converges to the minimum gap and reopens, or opens like a V;
  - bundle shifts (S-shaped steps), small undulations and long waves (smoothed noise), tears (missing pieces of a sheet);
  - bends: 1-3 successive shears (elbow, S or free curve, slope <= ~42 deg); each shear is a bijection, so sheets never cross;
  - a minimum gap GMIN between neighbouring sheets is enforced IN THE IMAGE (even inside bends).
Each sheet's median line is rasterised (1 px) with its own integer id ("axe_id"); the map is drawn with a margin and cropped,
so every line runs to the tile border. Sheet spacing (GMIN 9 px, base gap 14.5-19.5 px at ~9.4 um/px) was calibrated on real
m7 surface-prediction slices: 5th / 50th / 95th percentile spacing 4.75 / 7.75 / 16 px at 18.7 um/px.

Public API: random_settings(rng) -> dict, sheet_layout(settings, rng, size=(512, 512), margin=48) -> (H, W) uint16 sheet ids (0 = no line).
Internal names are French (original research code); see README for the settings table."""

import numpy as np, cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d
from scipy.special import beta as beta_fn
GMIN = 9.0
DU = 1.0
PAD = 220

def reglages(rng):
    u = rng.uniform
    return dict(gap=u(14.5, 19.5), gap_var=u(0.1, 0.25), paquet=u(5, 16), inter=u(1.3, 2.4), lent=u(1.2, 3.0), lent_amp=u(32, 70), lent_long=u(70, 200), lent_in=u(0.1, 0.4), pince=u(0.8, 2.0), decale=u(0.2, 1.0), ondule=u(0.5, 1.4), ondule_l=u(10, 25), vague=u(2, 9), vague_l=u(60, 160), vague_j=u(4, 15), dechir=u(0.25, 0.7), angle=u(0, np.pi), virage=u(0.4, 0.85), virage_l=u(70, 200), n_virages=int(rng.integers(1, 3)), ep=u(1.6, 3.2), gris_air=u(50, 60), gris_pap=u(120, 145), bruit=u(3, 7))

def _bruit(rng, NZ, K, nu, sz, sj, su, amp):
    f = max(1, int(su / 4))
    nn = nu // f + 4
    a = gaussian_filter(rng.normal(0, 1, (NZ, K, nn)), (sz if NZ > 1 else 0, sj, su / f), mode='nearest')
    a = a / (a.std() + 1e-09) * amp
    xs = np.linspace(0, nn - 1, nu)
    out = np.empty((NZ, K, nu), np.float32)
    i0 = np.clip(np.floor(xs).astype(int), 0, nn - 2)
    t = (xs - i0).astype(np.float32)
    for z in range(NZ):
        out[z] = a[z][:, i0] * (1 - t) + a[z][:, i0 + 1] * t
    return out

def _virages(R, rng, NZ, etendue):
    ag = np.arange(-etendue, etendue + 1, 2.0)
    ops = []
    zc = np.arange(NZ) - (NZ - 1) / 2
    for i in range(R['n_virages'] + (1 if etendue > 2000 else 0)):
        phi = R['angle'] + rng.normal(0, 0.2)
        s = R['virage'] * rng.uniform(0.6, 1.0) * (1.0 if i == 0 else 0.55) * rng.choice([-1, 1])
        w = R['virage_l'] * rng.uniform(0.7, 1.3)
        a0 = rng.uniform(-0.25, 0.25) * etendue / 2.5
        da = rng.normal(0, 0.6)
        kind = rng.choice(['coude', 'S', 'libre'], p=[0.4, 0.35, 0.25])
        if kind == 'libre':
            n = gaussian_filter1d(rng.normal(0, 1, ag.size), w / 2.0, mode='nearest')
            n = np.tanh(n / (n.std() + 1e-09) * 1.1)
        Fz = []
        for z in range(NZ):
            x = (ag - a0 - da * zc[z]) / w
            if kind == 'coude':
                dF = s * 0.5 * (1 + np.tanh(x))
            elif kind == 'S':
                dF = s * np.exp(-0.5 * x * x)
            else:
                dF = s * np.roll(n, int(round(da * zc[z] / 2)))
            F = np.cumsum(dF) * 2.0
            F -= np.interp(0.0, ag, F)
            Fz.append(F.astype(np.float64))
        ops.append((phi, ag, Fz))
    return ops

def _plier(u, v, theta, ops, z):
    c, s = (np.cos(theta), np.sin(theta))
    x = u * c - v * s
    y = u * s + v * c
    for phi, ag, Fz in ops:
        cp, sp = (np.cos(phi), np.sin(phi))
        a = x * cp + y * sp
        b = -x * sp + y * cp
        b = b + np.interp(a, ag, Fz[z])
        x = a * cp - b * sp
        y = a * sp + b * cp
    return (x, y)

def _deplier(x, y, theta, ops, z):
    for phi, ag, Fz in reversed(ops):
        cp, sp = (np.cos(phi), np.sin(phi))
        a = x * cp + y * sp
        b = -x * sp + y * cp
        b = b - np.interp(a, ag, Fz[z])
        x = a * cp - b * sp
        y = a * sp + b * cp
    c, s = (np.cos(theta), np.sin(theta))
    return (x * c + y * s, -x * s + y * c)

def _pile(R, rng, NZ, u, vmin, vmax, etire=None):
    nu = u.size
    zc = np.arange(NZ) - (NZ - 1) / 2
    g0, inter = ([], [])
    while sum(g0) < vmax - vmin + 2 * PAD:
        n = max(2, int(round(rng.gamma(2.0, R['paquet'] / 2))))
        b = R['gap'] * np.exp(rng.normal(0, 0.12))
        for _ in range(n - 1):
            g0.append(max(GMIN + 0.4, b * np.exp(rng.normal(0, R['gap_var']))))
            inter.append(False)
        g0.append(b * R['inter'] * np.exp(rng.normal(0, 0.3)))
        inter.append(True)
    g0 = np.array(g0)
    inter = np.array(inter)
    K = g0.size + 1
    Y = np.concatenate([[0.0], np.cumsum(g0)]) + vmin - PAD
    jc = int(np.argmin(np.abs(Y)))
    D = np.zeros((NZ, K, nu), np.float32)
    vides = []
    for j in range(K - 1):
        taux = R['lent'] if inter[j] else R['lent'] * R['lent_in']
        for _ in range(rng.poisson(taux * nu / 1000)):
            c = rng.uniform(u[0], u[-1])
            wl, wr = R['lent_long'] * np.exp(rng.normal(0, 0.35, 2))
            if not inter[j]:
                wl, wr = (wl * 0.6, wr * 0.6)
            A = min(R['lent_amp'] * np.exp(rng.normal(0, 0.45)), 0.6 * (wl + wr) / 2, 0.8 * min(wl, wr), 110.0) * (1.0 if inter[j] else 0.45)
            p = rng.uniform(1.5, 2.2)
            dc = rng.normal(0, 0.8)
            sig = rng.uniform(0.25, 0.75)
            lam = rng.uniform(8, 25)
            nuw = rng.uniform(3, 8)
            aire = beta_fn(0.5, p + 1) / beta_fn(0.5, 3.5)
            n_up = min(K - j - 1, int(3 * lam) + 1)
            n_dn = min(j + 1, int(3 * lam) + 1)
            nmax = max(n_up, n_dn)
            kmax = 1 + nmax / nuw
            marge_u = 2 * abs(dc) * NZ + 2
            i0 = int(np.searchsorted(u, c - wl * kmax - marge_u))
            i1 = int(np.searchsorted(u, c + wr * kmax + marge_u))
            if i1 <= i0:
                continue
            uu = u[i0:i1]
            n = np.arange(nmax)[:, None]
            kk = 1 + n / nuw
            for z in range(NZ):
                cz = c + dc * zc[z]
                x = np.where(uu[None] < cz, (uu[None] - cz) / (wl * kk), (uu[None] - cz) / (wr * kk))
                pr = np.clip(1 - x * x, 0, None)
                P = A * np.exp(-n / lam) * np.where(n == 0, pr ** p, aire * pr ** 2.5 / kk)
                D[z, j + 1:j + 1 + n_up, i0:i1] += sig * P[:n_up]
                D[z, j - n_dn + 1:j + 1, i0:i1] -= (1 - sig) * P[:n_dn][::-1]
            vides.append((j, c, wl, wr, A, p, dc))
    for _ in range(rng.poisson(R['pince'] * (nu / 1000) * (K / 100) * 2.5)):
        pince = rng.random() < 0.5
        m = int(rng.integers(4, 14)) if pince else int(rng.integers(3, 9))
        j1 = int(rng.integers(0, max(1, K - m)))
        j2 = min(K - 1, j1 + m)
        jm = (j1 + j2) // 2 if pince else j1 if rng.random() < 0.5 else j2
        relmax = max(Y[j2] - Y[jm], Y[jm] - Y[j1], 1.0)
        c = rng.uniform(u[0], u[-1])
        dc = rng.normal(0, 0.8)
        lam = rng.uniform(4, 14)
        if pince:
            o = min(rng.uniform(0.0, 1.0), 40 / relmax)
            m0 = rng.uniform(0.3, 0.5)
            amp = 1 + o - m0
            w = max(rng.uniform(40, 120), relmax * amp * 0.607 / 0.45)
        else:
            bmax = np.log(1 + 90 / relmax)
            beta = min(rng.uniform(0.6, 1.2), bmax) * rng.choice([-1, 1])
            w = max(rng.uniform(50, 140), relmax * abs(beta) * np.exp(abs(beta)) / 0.45)
        f = np.zeros((NZ, nu), np.float32)
        for z in range(NZ):
            x = (u - c - dc * zc[z]) / w
            f[z] = o - amp * np.exp(-0.5 * x * x) if pince else np.exp(beta * np.tanh(x)) - 1
        rel = (Y[j1:j2 + 1] - Y[jm]).astype(np.float32)
        D[:, j1:j2 + 1] += rel[None, :, None] * f[:, None, :]
        n_up = min(K - 1 - j2, int(4 * lam) + 1)
        if n_up > 0:
            D[:, j2 + 1:j2 + 1 + n_up] += rel[-1] * f[:, None, :] * np.exp(-(np.arange(n_up) + 1) / lam)[None, :, None]
        n_dn = min(j1, int(4 * lam) + 1)
        if n_dn > 0:
            D[:, j1 - n_dn:j1] += rel[0] * f[:, None, :] * np.exp(-(np.arange(n_dn)[::-1] + 1) / lam)[None, :, None]
    for _ in range(rng.poisson(R['decale'] * (nu / 1000) * (K / 100))):
        jb = rng.uniform(0, K)
        T = rng.uniform(2, 10)
        A = rng.uniform(15, 55) * rng.choice([-1, 1])
        c = rng.uniform(u[0], u[-1])
        w = max(rng.uniform(40, 120), abs(A) / 0.9)
        dc = rng.normal(0, 0.8)
        Wj = (1 / (1 + np.exp(-(np.arange(K) - jb) / T))).astype(np.float32)
        for z in range(NZ):
            D[z] += Wj[:, None] * (A * 0.5 * (1 + np.tanh((u - c - dc * zc[z]) / w)))[None, :].astype(np.float32)
    sz = 6.0
    D += _bruit(rng, NZ, K, nu, sz, R['vague_j'], R['vague_l'], R['vague'])
    D += _bruit(rng, NZ, K, nu, sz, 0.8, R['ondule_l'], R['ondule'])
    V = Y[None, :, None] + D
    G0 = np.diff(V, axis=1)
    V2 = V
    s = 0.6
    acc = np.full((NZ, K, nu), GMIN, np.float32)
    for _ in range(4 if etire is not None else 1):
        if etire is not None:
            for z in range(NZ):
                pente = np.gradient(gaussian_filter1d(V2[z], 4.0, axis=1, mode='nearest'), DU, axis=1)
                e = GMIN * etire(z, np.broadcast_to(u, V2[z].shape), V2[z], pente)
                e = gaussian_filter1d(maximum_filter1d(e, 25, axis=1, mode='nearest'), 6.0, axis=1, mode='nearest') * 1.03
                acc[z] = np.minimum(np.maximum(acc[z], e), 4 * GMIN)
        gm = np.maximum(acc[:, 1:], acc[:, :-1])
        G = gm + s * np.logaddexp(0, (G0 - gm) / s)
        V2 = np.concatenate([np.zeros((NZ, 1, nu), np.float32), np.cumsum(G, axis=1, dtype=np.float32)], axis=1)
        V2 += V[:, jc:jc + 1] - V2[:, jc:jc + 1]
    V2 = gaussian_filter1d(V2, 2.0, axis=2, mode='nearest')
    return (V2.astype(np.float32), vides, inter)

def _dechirures(R, rng, vides, u, V):
    trous = {}
    zm = V.shape[0] // 2
    for j, c, wl, wr, A, p, dc in vides:
        if A < 20 or rng.random() > R['dechir']:
            continue
        haut = rng.random() < 0.5
        jt = j + 1 if haut else j
        cote = -1.0 if haut else 1.0
        i0, i1 = (int(np.searchsorted(u, c - wl)), int(np.searchsorted(u, c + wr)))
        if i1 - i0 < 20:
            continue
        gap = V[zm, j + 1, i0:i1] - V[zm, j, i0:i1]
        im = int(np.argmax(gap))
        seuil = max(14.0, 0.5 * gap[im])
        if gap[im] < seuil:
            continue
        ouv = gap >= seuil
        a_ = im
        b_ = im
        while a_ > 0 and ouv[a_ - 1]:
            a_ -= 1
        while b_ < len(ouv) - 1 and ouv[b_ + 1]:
            b_ += 1
        lo, hi = (u[i0 + a_] + 3, u[i0 + b_] - 3)
        if hi - lo < 14:
            continue
        if rng.random() < 0.3:
            if rng.random() < 0.5:
                a, b = (u[0] - 10, rng.uniform(lo, hi))
            else:
                a, b = (rng.uniform(lo, hi), u[-1] + 10)
        else:
            a = rng.uniform(lo, hi - 6)
            b = a + rng.uniform(6, max(7, hi - a))
        trous.setdefault(jt, []).append((a, b, cote, j, dc, rng.uniform(0.03, 0.2), rng.uniform(15, 45)))
    return trous

def _decouper(P, W, H):
    lo, hx, hy = (-0.5, W - 0.5, H - 0.5)
    ins = (P[:, 0] >= lo) & (P[:, 0] <= hx) & (P[:, 1] >= lo) & (P[:, 1] <= hy)
    idx = np.nonzero(ins)[0]
    if not idx.size:
        return []

    def bord(pi, po):
        d = po - pi
        t = 1.0
        for k, (l, h) in enumerate(((lo, hx), (lo, hy))):
            if po[k] < l and d[k] != 0:
                t = min(t, (l - pi[k]) / d[k])
            if po[k] > h and d[k] != 0:
                t = min(t, (h - pi[k]) / d[k])
        return pi + max(0.0, t) * d
    out = []
    for r in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
        Q = P[r]
        if r[0] > 0:
            Q = np.vstack([bord(P[r[0]], P[r[0] - 1])[None], Q])
        if r[-1] < len(P) - 1:
            Q = np.vstack([Q, bord(P[r[-1]], P[r[-1] + 1])[None]])
        if len(Q) >= 2:
            out.append(Q.astype(np.float32))
    return out

def _pixels(Q, H, W):
    L = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    t = np.arange(0, L[-1] + 1e-06, 0.3)
    x = np.interp(t, L, Q[:, 0])
    y = np.interp(t, L, Q[:, 1])
    P = np.stack([np.clip(np.round(y), 0, H - 1), np.clip(np.round(x), 0, W - 1)], 1).astype(np.int32)
    P = P[np.concatenate([[True], np.any(np.diff(P, axis=0) != 0, axis=1)])]
    P = P.tolist()
    out = [P[0]]
    for p in P[1:]:
        while len(out) >= 2 and abs(p[0] - out[-2][0]) <= 1 and (abs(p[1] - out[-2][1]) <= 1):
            out.pop()
        if p != out[-1]:
            out.append(p)
    return np.array(out, np.int32)

def volume(R, rng, NZ=1, N=512, marge=48):
    H, W = (N, N) if np.isscalar(N) else (int(N[0]), int(N[1]))
    Hs, Ws = (H + 2 * marge, W + 2 * marge)
    etendue = 1.5 * np.hypot(Hs, Ws) + 400
    ops = _virages(R, rng, NZ, etendue)
    theta = R['angle']
    t = np.arange(0, 1, 2.0 / (Hs + Ws))
    bx = np.concatenate([t * Ws, np.full_like(t, Ws), (1 - t) * Ws, np.zeros_like(t)]) - Ws / 2
    by = np.concatenate([np.zeros_like(t), t * Hs, np.full_like(t, Hs), (1 - t) * Hs]) - Hs / 2
    U, Vv = ([], [])
    for z in range(NZ):
        uu, vv = _deplier(bx, by, theta, ops, z)
        U.append(uu)
        Vv.append(vv)
    U, Vv = (np.concatenate(U), np.concatenate(Vv))
    u = np.arange(U.min() - 30, U.max() + 30, DU).astype(np.float32)

    def etire(z, uu, vv, pente, h=0.5):
        x1, y1 = _plier(uu, vv, theta, ops, z)
        x2, y2 = _plier(uu + h, vv + h * pente, theta, ops, z)
        return (np.hypot(x2 - x1, y2 - y1) / h).astype(np.float32)
    V, vides, inter = _pile(R, rng, NZ, u, float(Vv.min()), float(Vv.max()), etire)
    trous = _dechirures(R, rng, vides, u, V)
    K = V.shape[1]
    ep = np.clip(R['ep'] * np.exp(rng.normal(0, 0.25, K)), 1, 4)
    morceaux = [[] for _ in range(NZ)]
    vus = set()
    bord_touche = False
    zc = np.arange(NZ) - (NZ - 1) / 2
    for z in range(NZ):
        for j in range(K):
            v = V[z, j].copy()
            ok = np.ones(u.size, bool)
            bouts = []
            for a, b, cote, jv, dc, dlt, ld in trous.get(j, []):
                a2, b2 = (a + dc * zc[z], b + dc * zc[z])
                ok &= ~((u > a2) & (u < b2))
                bouts += [(a2, cote, jv, dlt, ld), (b2, cote, jv, dlt, ld)]
            for ue, cote, jv, dlt, ld in bouts:
                if ue < u[0] or ue > u[-1]:
                    continue
                i = int(np.clip(np.searchsorted(u, ue), 0, u.size - 1))
                larg = V[z, jv + 1, i] - V[z, jv, i]
                v += ok * cote * dlt * max(0.0, larg - GMIN) * np.clip(1 - np.abs(u - ue) / ld, 0, None) ** 2
            idx = np.nonzero(ok)[0]
            if not idx.size:
                continue
            for seg in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                if len(seg) < 25:
                    continue
                x, y = _plier(u[seg].astype(np.float64), v[seg].astype(np.float64), theta, ops, z)
                P = np.stack([x + W / 2 - 0.5, y + H / 2 - 0.5], 1)
                for Q in _decouper(P, W, H):
                    morceaux[z].append((j, Q))
                    vus.add(j)
                    if j in (0, K - 1):
                        bord_touche = True
    num = {j: i + 1 for i, j in enumerate(sorted(vus))}
    axe_id = np.zeros((NZ, H, W), np.uint16)
    feuille = np.zeros((NZ, H, W), np.uint16)
    dens = np.zeros((NZ, H, W), np.float32)
    lignes = []
    for z in range(NZ):
        Lz = []
        for j, Q in morceaux[z]:
            k = num[j]
            Lz.append((k, Q))
            q = np.round(Q * 16).astype(np.int32)
            px = _pixels(Q, H, W)
            axe_id[z][px[:, 0], px[:, 1]] = k
            e = max(1, int(round(ep[j])))
            cv2.polylines(feuille[z], [q], False, int(k), e, cv2.LINE_8, 4)
            cv2.polylines(dens[z], [q], False, 1.0, e, cv2.LINE_AA, 4)
        lignes.append(Lz)
    img = gaussian_filter(dens, (0, 0.8, 0.8))
    img = R['gris_air'] + (R['gris_pap'] - R['gris_air']) * np.clip(img, 0, 1.2)
    img += gaussian_filter(rng.normal(0, 1, img.shape).astype(np.float32), (0, 1.2, 1.2)) * R['bruit'] * 3
    return dict(axe_id=axe_id, axe=(axe_id > 0).astype(np.uint8), lignes=lignes, papyrus=(dens > 0.4).astype(np.uint8), feuille=feuille, ct=np.clip(img, 0, 255).astype(np.uint8), couverture_ok=not bord_touche, n_pile=K, plan=dict(u=u, V=V, numero=num, theta=theta, ops=ops))

def controle(axe_id, lignes, vide_px=15.0):
    from scipy.ndimage import distance_transform_edt, label
    from scipy.spatial import cKDTree
    H, W = axe_id.shape
    cnt = np.bincount(axe_id.ravel())
    n_f = int((cnt[1:] >= 10).sum())
    d, ind = distance_transform_edt(axe_id == 0, return_indices=True)
    lab = axe_id[ind[0], ind[1]]
    g = []
    for A_, B_ in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))), ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        m = lab[A_] != lab[B_]
        g.append((d[A_] + d[B_] + 1)[m])
    g = np.concatenate(g) if g else np.zeros(1)
    vide = float((d > vide_px).mean())
    bouts, bouts_vide = (0, 0)
    for k, Q in lignes:
        for p in (Q[0], Q[-1]):
            if min(p[0] + 0.5, W - 0.5 - p[0], p[1] + 0.5, H - 0.5 - p[1]) <= 2.0:
                continue
            bouts += 1
            x0, y0 = (int(round(p[0])), int(round(p[1])))
            if d[max(0, y0 - 10):y0 + 11, max(0, x0 - 10):x0 + 11].max() >= 7:
                bouts_vide += 1
    A, B, S, M, I = ([], [], [], [], [])
    for n, (k, Q) in enumerate(lignes):
        A.append(Q[:-1])
        B.append(Q[1:])
        S.append(np.full(len(Q) - 1, k))
        M.append(np.full(len(Q) - 1, n))
        I.append(np.arange(len(Q) - 1))
    A, B, S, M, I = (np.concatenate(x) for x in (A, B, S, M, I))
    lmax = float(np.linalg.norm(B - A, axis=1).max())
    arbre_s = cKDTree((A + B) / 2)
    pr = arbre_s.query_pairs(lmax + 0.001, output_type='ndarray')
    pr = pr[~((M[pr[:, 0]] == M[pr[:, 1]]) & (np.abs(I[pr[:, 0]] - I[pr[:, 1]]) <= 1))]

    def orient(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
    p1, p2, q1, q2 = (A[pr[:, 0]], B[pr[:, 0]], A[pr[:, 1]], B[pr[:, 1]])
    coupe = (orient(p1, p2, q1) * orient(p1, p2, q2) < 0) & (orient(q1, q2, p1) * orient(q1, q2, p2) < 0)
    croisements = int((coupe & (S[pr[:, 0]] != S[pr[:, 1]])).sum())
    boucles_geo = int((coupe & (S[pr[:, 0]] == S[pr[:, 1]])).sum())
    P = np.concatenate([Q for _, Q in lignes])
    Pk = np.concatenate([np.full(len(Q), k) for k, Q in lignes])
    pp = cKDTree(P).query_pairs(3.0, output_type='ndarray')
    pp = pp[Pk[pp[:, 0]] != Pk[pp[:, 1]]]
    dmin = float(np.linalg.norm(P[pp[:, 0]] - P[pp[:, 1]], axis=1).min()) if len(pp) else 3.0
    Z = np.pad(axe_id.astype(np.int32), 1)
    c0 = Z[1:-1, 1:-1]
    tour = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    nb = [(Z[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] == c0) & (c0 > 0) for dy, dx in tour]
    passages = sum((~nb[i] & nb[(i + 1) % 8] for i in range(8)))
    fourches = int(((c0 > 0) & (passages >= 3)).sum())
    trous = 0
    for k in np.unique(axe_id[axe_id > 0]):
        yy, xx = np.nonzero(axe_id == k)
        y0, y1, x0, x1 = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)
        m = np.pad(axe_id[y0:y1, x0:x1] == k, 1)
        trous += label(~m)[1] - 1
    return dict(n_feuilles=n_f, ecart_median=float(np.median(g)), ecart_p10=float(np.percentile(g, 10)), ecart_p90=float(np.percentile(g, 90)), part_vide=vide, bouts_interieurs=bouts, bouts_dans_un_vide=bouts_vide, croisements=croisements, fourches=fourches, boucles=int(boucles_geo + trous), distance_min_entre_feuilles=dmin)


def random_settings(rng):
    """Draw one random set of layout settings (see README)."""
    return reglages(rng)


def sheet_layout(settings, rng, size=(512, 512), margin=48):
    """(H, W) uint16 map: 0 = background, k > 0 = median line (1 px) of sheet k."""
    import numpy as _np
    ids = _np.asarray(volume(settings, rng, 1, size, margin)["axe_id"])
    return (ids[0] if ids.ndim == 3 else ids).astype(_np.uint16)
