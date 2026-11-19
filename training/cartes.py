"""CARTES DE FEUILLES CONTINUES : des lignes continues, pas coupées comme les cartes m7 brutes (principe du jeu de données).
  mesures(sk)  : boucles fermées (trous enclos de 10 à 400 px), nœuds intérieurs (extrémités loin du bord + fourches, pour 1 000 px de trait),
                 part des traits courts (< 60 px), continuité (part du trait dans des courbes de ≥ 200 px ou allant d'un bord à l'autre) ;
  rejet(m)     : carte REJETÉE ENTIÈRE (papyrus broyé / m7 en taches = on ignore) si boucle(s), nœuds trop denses ou traits courts ;
  reparer(sk, ct) : pour une carte acceptée seulement : petits trous de squelettisation, ébarbage des bouts < 8 px, raccord des coupures
                 (≤ 30 px, dans l'orientation locale ± 35°) SEULEMENT si le scan montre la feuille dans le trou (≥ 80 % du pont sur du papyrus),
                 sinon la coupure reste (vraie déchirure) ; puis morceaux < 30 px retirés.
Seuils réglés sur la répartition des 33 697 paires (journal)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu

K3 = np.ones((3, 3), bool); U3 = np.ones((3, 3), np.uint8)
# Seuils calés sur les étiquettes exactes de Paris 4 (nœuds médiane 9,4 / 75e centile 13,9 ; continuité médiane 0,90 / 25e centile 0,84).
SEUILS = dict(boucles=0, noeuds=14.0, courts=0.20, cont=0.80, formes=0.05)


def papyrus(ct):
    g = ndi.gaussian_filter(ct.astype(np.float32), 1.0); m = g > threshold_otsu(g)
    m = ndi.binary_opening(m, structure=ndi.generate_binary_structure(2, 1)); return remove_small_objects(m, 40)


def _nb(sk): return ndi.convolve(sk.astype(np.uint8), U3, mode="constant") - sk.astype(np.uint8)


def _propre(sk):
    sk = sk.astype(bool)
    if not sk.any(): return sk
    # débruitage du squelette AVANT toute mesure (pas une réparation) : trous de 1 à 9 px et barbules de ≤ 6 px = artefacts de squelettisation
    return _ebarber(skeletonize(remove_small_holes(sk, 10)), 6)


def mesures(sk, marge=4):
    sk = _propre(sk); n = int(sk.sum()); H, W = sk.shape
    if n == 0: return dict(px=0, boucles=0, noeuds=0.0, courts=1.0, cont=0.0), sk
    nb = _nb(sk); ep = sk & (nb == 1); jn = sk & (nb >= 3)
    inter = np.zeros_like(sk); inter[marge:H - marge, marge:W - marge] = True
    fond, nf = ndi.label(~sk); tf = np.bincount(fond.ravel(), minlength=nf + 1)
    bord = set(np.unique(np.r_[fond[0], fond[-1], fond[:, 0], fond[:, -1]]).tolist())
    boucles = sum(1 for i in range(1, nf + 1) if tf[i] >= 10 and i not in bord)      # TOUTE courbe fermée (consigne : aucune)
    n_jn = ndi.label(jn, structure=K3)[1]; n_ep = int((ep & inter).sum())
    lab, nc = ndi.label(sk, structure=K3); sz = np.bincount(lab.ravel(), minlength=nc + 1)
    ne = np.bincount(lab[ep], minlength=nc + 1); neb = np.bincount(lab[ep & ~inter], minlength=nc + 1)
    ok = (sz >= 200) | ((ne > 0) & (ne == neb)); ok[0] = False
    # « formes » : virages en épingle et traits parallèles collés (< 4-5 px) = trait localement étalé en 2D (ACP locale, disque de rayon 5 px)
    r = 4; yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; Kd = (yy ** 2 + xx ** 2 <= r * r).astype(np.float32); s = sk.astype(np.float32)
    N = ndi.convolve(s, Kd, mode="constant") + 1e-6; mx = ndi.convolve(s, Kd * xx, mode="constant") / N; my = ndi.convolve(s, Kd * yy, mode="constant") / N
    cxx = ndi.convolve(s, Kd * xx * xx, mode="constant") / N - mx ** 2; cyy = ndi.convolve(s, Kd * yy * yy, mode="constant") / N - my ** 2
    cxy = ndi.convolve(s, Kd * xx * yy, mode="constant") / N - mx * my
    tr_, dt_ = cxx + cyy, cxx * cyy - cxy ** 2; disc = np.sqrt(np.maximum(tr_ ** 2 / 4 - dt_, 0)); l1, l2 = tr_ / 2 + disc, tr_ / 2 - disc
    loin = sk & ~ndi.binary_dilation(jn, structure=K3, iterations=3); loin[:r, :] = loin[-r:, :] = loin[:, :r] = loin[:, -r:] = False
    formes = float(((l2 / np.maximum(l1, 1e-6)) > 0.12)[loin].mean()) if loin.any() else 0.0
    return dict(px=n, boucles=int(boucles), noeuds=float((n_ep + n_jn) / n * 1000), courts=float(sz[1:][sz[1:] < 60].sum() / n),
                cont=float(sz[ok].sum() / n), formes=formes), sk


def rejet(m, S=SEUILS):
    r = []
    if m["boucles"] > S["boucles"]: r.append("boucles")
    if m["noeuds"] > S["noeuds"]: r.append("noeuds")
    if m["courts"] > S["courts"]: r.append("courts")
    if m.get("formes", 0) > S["formes"]: r.append("formes")
    if m["px"] < 150: r.append("vide")
    return r


def _ebarber(sk, n=8):
    s = sk.copy()
    for _ in range(n): s &= ~(s & (_nb(s) == 1))
    g = s & (_nb(s) == 1)
    for _ in range(n): g = ndi.binary_dilation(g, structure=K3) & sk
    return s | g


def _direction(sk, p, n=10):
    """direction du trait à son extrémité p (vecteur sortant), en remontant n px"""
    y, x = p; vu = {(y, x)}; cur = (y, x)
    for _ in range(n):
        vs = [(cur[0] + dy, cur[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)]
        vs = [v for v in vs if 0 <= v[0] < sk.shape[0] and 0 <= v[1] < sk.shape[1] and sk[v] and v not in vu]
        if len(vs) != 1: break
        cur = vs[0]; vu.add(cur)
    d = np.array([y - cur[0], x - cur[1]], np.float32); nn = np.linalg.norm(d)
    return d / nn if nn >= 3 else None


def reparer(sk, ct, R=30, ang=35.0, pap=None, bouts_seulement=False):
    import cv2
    from scipy.spatial import cKDTree
    sk = _ebarber(_propre(sk)); H, W = sk.shape
    if not sk.any(): return sk, 0, 0
    pap = papyrus(ct) if pap is None else pap; papd = ndi.binary_dilation(pap, iterations=1)
    lab, nc = ndi.label(sk, structure=K3); par = list(range(nc + 1))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    ep = np.argwhere(sk & (_nb(sk) == 1)); cmin = np.cos(np.radians(ang))
    # v3 (zéro branche) : on ne raccorde qu'un BOUT à un autre BOUT qui lui fait face (sinon le pont créerait une fourche en T)
    pts = ep if bouts_seulement else np.argwhere(sk)
    if len(pts) == 0: return sk, 0, 0
    arbre = cKDTree(pts); dirs = {tuple(q): _direction(sk, tuple(q)) for q in ep} if bouts_seulement else {}
    ponts = refus = 0; out = sk.astype(np.uint8)
    for p in ep:
        if min(p[0], p[1], H - 1 - p[0], W - 1 - p[1]) < 4: continue          # bout au bord de la tuile : fin naturelle
        d = _direction(sk, tuple(p))
        if d is None: continue
        best = None
        for j in arbre.query_ball_point(p, R):
            q = pts[j]; v = (q - p).astype(np.float32); L = float(np.hypot(*v))
            if L < 2 or f(lab[tuple(q)]) == f(lab[tuple(p)]): continue
            c = float(v @ d) / L
            if c < cmin: continue
            if bouts_seulement:
                dq = dirs.get(tuple(q))
                if dq is None or float(-v @ dq) / L < cmin: continue                 # l'autre bout doit pointer vers celui-ci
            s = L / c
            if best is None or s < best[0]: best = (s, q, L)
        if best is None: continue
        q = best[1]; n = int(best[2]) + 1
        yy = np.round(np.linspace(p[0], q[0], n)).astype(int); xx = np.round(np.linspace(p[1], q[1], n)).astype(int)
        if papd[yy, xx].mean() < 0.8: refus += 1; continue                    # le scan ne montre pas la feuille dans le trou : vraie coupure
        cv2.line(out, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, 1); par[f(lab[tuple(p)])] = f(lab[tuple(q)]); ponts += 1
    out = skeletonize(out > 0); out = remove_small_objects(out, 30, connectivity=2)
    return out, ponts, refus


# ================================================= v3 : ZÉRO BRANCHE, LIGNES LISSÉES, COLLÉES À LA CRÊTE ==========================================
# règle : ligne continue, pas de branche, pas de petites formes en éclair
#   1. après ébarbage des épines < 8 px, AUCUNE fourche (sinon tuile rejetée) ; chaque feuille = une courbe simple d'un bout à l'autre ;
#   2. chaque ligne lissée le long d'elle-même (gaussienne σ 8 px sur les coordonnées), puis contrôle : tortuosité (longueur / corde) sur 30 px
#      < 1,03 et aucun changement d'orientation > 25° sur 10 px ;
#   3. après lissage la ligne reste sur sa feuille : ≤ 2 px du centre de la crête claire du scan sur ≥ 90 % de sa longueur ;
#   4. mêmes contrôles (sauf la crête) pour les cartes du simulateur.
V3 = dict(sigma=8.0, tort=1.03, fen_tort=30, virage=25.0, fen_virage=10, crete_px=2.0, crete_part=0.90, cont=0.70, long_min=30)
OFFS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def fourches(sk):
    """vraies fourches : amas de pixels à ≥ 3 voisins d'où partent ≥ 3 branches distinctes (les coins d'escalier n'en sont pas)"""
    sk = sk.astype(bool); nb = _nb(sk); jn = sk & (nb >= 3)
    if not jn.any(): return 0
    jl, nj = ndi.label(jn, structure=K3); br = sk & ~jn; bl, _ = ndi.label(br, structure=K3); n = 0
    for j, sl in enumerate(ndi.find_objects(jl), 1):
        y0, y1, x0, x1 = max(0, sl[0].start - 1), sl[0].stop + 1, max(0, sl[1].start - 1), sl[1].stop + 1
        z = ndi.binary_dilation(jl[y0:y1, x0:x1] == j, structure=K3) & br[y0:y1, x0:x1]
        if len(np.unique(bl[y0:y1, x0:x1][z])) >= 3: n += 1
    return n


def chemin(comp):
    """pixels d'une composante sans fourche -> suite ordonnée (y, x) d'un bout à l'autre ; None si boucle ou pas une courbe simple"""
    P = set(map(tuple, np.argwhere(comp)))
    nbv = lambda c: [(c[0] + a, c[1] + b) for a, b in OFFS if (c[0] + a, c[1] + b) in P]
    bouts = [c for c in P if len(nbv(c)) == 1]
    if not bouts: return None
    path = [bouts[0]]; vu = {bouts[0]}
    while True:
        cur = path[-1]; cand = [c for c in nbv(cur) if c not in vu]
        if not cand: break
        c4 = [c for c in cand if c[0] == cur[0] or c[1] == cur[1]]; nx = (c4 or cand)[0]; path.append(nx); vu.add(nx)
    if len(vu) < 0.9 * len(P): return None
    return np.array(path, np.float32)


def _reech(c, pas=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    if d[-1] < 2: return c
    s = np.arange(0, d[-1] + 1e-6, pas); return np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], 1)


def lisser(c, sigma=None):
    from scipy.ndimage import gaussian_filter1d
    sigma = V3["sigma"] if sigma is None else sigma; c = _reech(c)
    if len(c) < 5: return c
    return _reech(np.stack([gaussian_filter1d(c[:, 0], sigma, mode="nearest"), gaussian_filter1d(c[:, 1], sigma, mode="nearest")], 1))


def controle_courbe(c, g=None, H=256, W=256):
    """-> liste des défauts : « zigzag » (tortuosité), « virage », « decollee » (loin de la crête du scan g lissé)"""
    r = []; n = len(c); ft, fv = V3["fen_tort"], V3["fen_virage"]
    if n > ft:
        corde = np.hypot(*(c[ft:] - c[:-ft]).T)
        if (ft / np.maximum(corde, 1e-6)).max() > V3["tort"]: r.append("zigzag")
    if n > fv + 6:
        t = c[6:] - c[:-6]; t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        cs = np.clip((t[fv:] * t[:-fv]).sum(1), -1, 1)
        if np.degrees(np.arccos(cs)).max() > V3["virage"]: r.append("virage")
    if g is not None and n > 6:
        from scipy.ndimage import map_coordinates
        t = np.gradient(c, axis=0); t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9); nn = np.stack([-t[:, 1], t[:, 0]], 1)
        dans = (c[:, 0] > 4) & (c[:, 0] < H - 5) & (c[:, 1] > 4) & (c[:, 1] < W - 5)
        if dans.sum() > 5:
            o = np.arange(-4, 4.01, 0.5); cc, nv = c[dans], nn[dans]
            prof = np.stack([map_coordinates(g, [cc[:, 0] + k * nv[:, 0], cc[:, 1] + k * nv[:, 1]], order=1, mode="nearest") for k in o], 1)
            dec = np.abs(o[prof.argmax(1)])
            if (dec <= V3["crete_px"]).mean() < V3["crete_part"]: r.append("decollee")
    return r


def tracer(courbes, H=256, W=256):
    import cv2
    out = np.zeros((H, W), np.uint8)
    for c in courbes:
        pts = np.round(c[:, ::-1]).astype(np.int32)
        if len(pts) > 1: cv2.polylines(out, [pts], False, 1, 1, cv2.LINE_8)
    return out > 0


def carte_v3(sk, ct=None, chemins=None, sigma=None, reference="crete", rogner=0):
    """carte -> (carte finale 1 px, motifs de rejet, infos). sk : carte brute (m7, simulateur) ; ou chemins : courbes déjà
    ordonnées (Paris 4 : ligne médiane la plus longue de chaque feuille). ct : coupe du scan (raccords + contrôle de crête)."""
    info = dict(ponts=0, refus=0)
    if chemins is None:
        m0, s0 = mesures(sk); r = rejet(m0)
        if r: return s0, r, info
        s0 = _ebarber(s0, 8)
        if ct is not None: s0, info["ponts"], info["refus"] = reparer(s0, ct, bouts_seulement=True)
        s0 = remove_small_objects(s0.astype(bool), V3["long_min"], connectivity=2)
        if fourches(s0): return s0, ["fourche"], info
        lab, nc = ndi.label(s0, structure=K3); chemins = []
        for j, sl in enumerate(ndi.find_objects(lab), 1):
            p = chemin(lab[sl] == j)
            if p is None: return s0, ["boucles"], info
            chemins.append(p + np.array([sl[0].start, sl[1].start], np.float32))
    # reference = « crete » : le trait lissé doit rester sur la crête claire du scan (cartes m7, approchées) ;
    #             « trace » : il doit rester à ≤ 2 px du tracé exact d'origine (Paris 4, simulateur : la carte est la vérité ;
    #             sur Paris 4 la crête la plus claire est souvent une des 2 couches de fibres, à 2–4 px du milieu de la feuille)
    g = ndi.gaussian_filter(ct.astype(np.float32), 1.5) if (ct is not None and reference == "crete") else None
    lisses, r = [], set()
    for p in chemins:
        if rogner and len(p) >= 2 * rogner + V3["long_min"]: p = p[rogner:-rogner]       # bouts du squelette d'une feuille pleine = crochets vers les coins
        if len(p) < V3["long_min"]: continue
        c = lisser(p, sigma); r |= set(controle_courbe(c, g))
        if reference == "trace":
            from scipy.spatial import cKDTree
            dd, _ = cKDTree(p).query(c)
            if (dd <= V3["crete_px"]).mean() < V3["crete_part"]: r.add("decollee")
        lisses.append(c)
    fin = tracer(lisses)
    if r: return fin, sorted(r), info
    m1, fin = mesures(fin)
    if fourches(fin) or m1["boucles"]: return fin, ["contact"], info                 # deux lignes lissées qui se touchent
    if m1["px"] < 150: return fin, ["vide"], info
    if m1["cont"] < V3["cont"]: return fin, ["continuite"], info
    info.update(m1); return fin, [], info


def lignes_paris4(l, long_min=30):
    """étiquettes d'une coupe -> pour chaque feuille (instance), son plus long chemin médian (squelette de la feuille bouchée)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    out = []
    for i in np.unique(l):
        if i == 0: continue
        r = l == i
        if r.sum() < 30: continue
        r = ndi.binary_fill_holes(ndi.binary_closing(r, iterations=2) | r); s = skeletonize(r)
        P = np.argwhere(s)
        if len(P) < long_min: continue
        idx = -np.ones(s.shape, np.int64); idx[P[:, 0], P[:, 1]] = np.arange(len(P)); a, b, w = [], [], []
        for dy, dx in OFFS:
            q = P + [dy, dx]; ok = (q[:, 0] >= 0) & (q[:, 0] < s.shape[0]) & (q[:, 1] >= 0) & (q[:, 1] < s.shape[1])
            k = idx[q[ok, 0], q[ok, 1]]; m = k >= 0; a += list(np.nonzero(ok)[0][m]); b += list(k[m]); w += [float(np.hypot(dy, dx))] * int(m.sum())
        Gm = coo_matrix((w, (a, b)), shape=(len(P), len(P))).tocsr()
        d0 = shortest_path(Gm, indices=0, directed=False); d0[~np.isfinite(d0)] = -1; u = int(d0.argmax())
        du, pr = shortest_path(Gm, indices=u, directed=False, return_predecessors=True); du[~np.isfinite(du)] = -1; v = int(du.argmax())
        ch = [v]
        while ch[-1] != u and pr[ch[-1]] >= 0: ch.append(int(pr[ch[-1]]))
        if len(ch) >= long_min: out.append(P[ch].astype(np.float32))
    return out
