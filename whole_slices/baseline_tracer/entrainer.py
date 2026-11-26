"""25/11 · PREMIER ENTRAÎNEMENT DU TRACEUR (accord de Vincent le 25/11 au soir). Apprend sur les coupes synthétiques (carrés de 512 px tirés au hasard) à sortir :
  - la LIGNE des feuilles (carte douce autour du tracé de vérité, continu à travers les cassures) ;
  - la PHASE entre deux feuilles (0 sur la feuille, tourne régulièrement jusqu'à la feuille suivante), sous forme sinus / cosinus.
Les nouvelles coupes préparées (preparer.py) sont prises au fil de l'eau. Tous les EVAL pas : contrôle sur les TÉMOINS (jamais vus) = 4 synthétiques + Paris 4.
    python entrainer.py <dossier de travail> [pas_max=60000] [lot=16]      reprend <dossier>/dernier.pt s'il existe ; s'arrête quand <dossier>/STOP_ENTR existe
Sorties : <dossier>/journal_entr.txt, eval.jsonl, planches/, dernier.pt, meilleur.pt (meilleure note Paris 4)."""
import sys, os, json, time, glob, math, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
D = sys.argv[1]; PAS_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 60000; LOT = int(sys.argv[3]) if len(sys.argv) > 3 else 16; T = 512; EVAL = int(os.environ.get("EVAL", 1500)); dev = "cuda"
TEMOINS = set(os.environ.get("TEMOINS", "c1496,c1497,c1498,c1499").split(",")); os.makedirs(D + "/planches", exist_ok=True)
def note(s):
    print(time.strftime("%H:%M:%S"), s, flush=True); open(D + "/journal_entr.txt", "a", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
# ---------------------------------------------------------------- données
class Coupes(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = np.random.default_rng((info.id if info else 0) * 9973 + int(time.time()) % 100000); liste = []; reels = []; t_liste = 0; cache = {}; centres_ = {}; P_REEL = float(os.environ.get("P_REEL", 0.25)); cartes = {}; ecrasees = set(); P_DENSE = float(os.environ.get("P_DENSE", 0.85))
        while True:
            if time.time() - t_liste > 300:                                                    # reprend la liste : de nouvelles coupes arrivent
                for j in glob.glob(D + "/prep/[cr]*.json"):
                    if j[:-5] + ".npy" not in centres_:
                        try:
                            m_ = json.load(open(j))
                            if "centre" in m_:
                                centres_[j[:-5] + ".npy"] = m_["centre"]
                                if m_.get("famille") == "ecrase": ecrasees.add(j[:-5] + ".npy")
                        except Exception: pass
                liste = [p for p in sorted(glob.glob(D + "/prep/c*.npy")) if os.path.basename(p)[:-4] not in TEMOINS and not p.endswith("_tmp.npy") and p in centres_]; reels = [p for p in sorted(glob.glob(D + "/prep/r*.npy")) if not p.endswith("_tmp.npy") and p in centres_]; t_liste = time.time()
            if not liste: time.sleep(10); t_liste = 0; continue
            reel = bool(reels) and rng.random() < P_REEL; p = reels[rng.integers(len(reels))] if reel else liste[rng.integers(len(liste))]
            if not reel and p not in ecrasees and rng.random() < 0.5: p = liste[rng.integers(len(liste))]                 # deuxième tirage si la coupe est lâche : les écrasées sortent ~2 fois plus
            if p not in cache:
                if len(cache) > 24: cache.pop(next(iter(cache)))
                cache[p] = np.load(p, mmap_mode="r")
            a = cache[p]; _, H, W = a.shape; e = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))); S = int(round(T * e))   # changement d'échelle : couvre 7 à 11 µm
            if H <= S + 2 or W <= S + 2: continue
            if p not in cartes:                                                                    # carte grossière de la densité de feuilles (blocs de 64 px), calculée une fois par coupe
                if len(cartes) > 48: cartes.pop(next(iter(cartes)))
                hb, wb = H // 64, W // 64; lg_ = np.asarray(a[1][:hb * 64, :wb * 64]).reshape(hb, 64, wb, 64).mean((1, 3)) / 255.0; vd_ = np.asarray(a[3][:hb * 64, :wb * 64]).reshape(hb, 64, wb, 64).mean((1, 3)) / 255.0
                dn_ = cv2.blur(lg_.astype(np.float32), (7, 7)) * (vd_ > 0.5); pr_ = dn_.ravel() ** 2; cartes[p] = (pr_ / pr_.sum() if pr_.sum() > 0 else None, wb)
            pr_, wb = cartes[p]
            if pr_ is not None and rng.random() < P_DENSE:                                           # carré tiré là où les feuilles sont serrées
                k_ = int(rng.choice(len(pr_), p=pr_)); cy_, cx_ = (k_ // wb) * 64 + int(rng.integers(64)), (k_ % wb) * 64 + int(rng.integers(64)); y, x = int(np.clip(cy_ - S // 2, 0, H - S - 1)), int(np.clip(cx_ - S // 2, 0, W - S - 1))
            else:
                for _ in range(8):
                    y, x = int(rng.integers(0, H - S)), int(rng.integers(0, W - S))
                    if a[3, y + S // 2, x + S // 2] > 0 or rng.random() < 0.06: break
            c = np.array(a[:, y:y + S, x:x + S]); img, lig, ph, ded = c[0], c[1] > 0, c[2], c[3] > 0; cas = cv2.dilate((c[4] > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
            d = np.minimum(cv2.distanceTransform((~lig).astype(np.uint8), cv2.DIST_L2, 3), 64.0) / e                 # distance au tracé, en pixels de la sortie
            ang = ph.astype(np.float32) * (2 * np.pi / 256); si, co = np.sin(ang), np.cos(ang)
            if S != T:
                img = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA if S > T else cv2.INTER_CUBIC); d = cv2.resize(d, (T, T), interpolation=cv2.INTER_LINEAR)
                si, co = cv2.resize(si, (T, T), interpolation=cv2.INTER_LINEAR), cv2.resize(co, (T, T), interpolation=cv2.INTER_LINEAR); ded = cv2.resize(ded.astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST) > 0; cas = cv2.resize(cas, (T, T), interpolation=cv2.INTER_NEAREST)
            x_ = img.astype(np.float32) / 255.0
            # variations d'aspect : ton, contraste, flou, bruit (le réel n'a jamais exactement notre ton)
            x_ = np.clip(x_, 0, 1) ** float(np.exp(rng.normal(0, 0.25))); x_ = (x_ - 0.3) * float(np.exp(rng.normal(0, 0.2))) + 0.3 + float(rng.normal(0, 0.05))
            if rng.random() < 0.4: x_ = cv2.GaussianBlur(x_, (0, 0), float(rng.uniform(0.4, 1.3)))
            if rng.random() < 0.5: x_ = x_ + rng.normal(0, float(rng.uniform(0.005, 0.04)), x_.shape).astype(np.float32)
            cible = np.stack([np.exp(-(d / 1.6) ** 2), si, co, ded.astype(np.float32), cas, np.full((T, T), float(reel), np.float32)]).astype(np.float32)
            cx, cy = centres_[p]; X = x + (np.arange(T, dtype=np.float32) + 0.5) * e; Y = y + (np.arange(T, dtype=np.float32) + 0.5) * e; r = np.hypot(X[None, :] - cx, Y[:, None] - cy)
            x_ = np.stack([x_, (r - r.mean()) / 256.0, r / 3000.0]).astype(np.float32)                    # image + distance au centre (rampe locale, valeur absolue)
            k = int(rng.integers(4)); x_, cible = np.rot90(x_, k, (1, 2)), np.rot90(cible, k, (1, 2))
            if rng.random() < 0.5: x_, cible = x_[:, :, ::-1], cible[:, :, ::-1]
            yield torch.from_numpy(np.ascontiguousarray(x_)), torch.from_numpy(np.ascontiguousarray(cible))
# ---------------------------------------------------------------- réseau (U-Net)
class Bloc(nn.Module):
    def __init__(s, a, b):
        super().__init__(); s.c1 = nn.Conv2d(a, b, 3, padding=1); s.n1 = nn.GroupNorm(8, b); s.c2 = nn.Conv2d(b, b, 3, padding=1); s.n2 = nn.GroupNorm(8, b); s.sk = nn.Conv2d(a, b, 1) if a != b else nn.Identity()
    def forward(s, x): h = F.silu(s.n1(s.c1(x))); h = s.n2(s.c2(h)); return F.silu(h + s.sk(x))
class Traceur(nn.Module):
    def __init__(s, C=None):
        C = C or tuple(int(v) for v in os.environ.get("LARGEURS", "32,64,128,256,384").split(","))                    # 26/11 : largeurs et nombre d'étages réglables (essai d'un réseau plus gros, qui voit plus large)
        super().__init__(); s.bas = nn.ModuleList(); s.haut = nn.ModuleList(); a = 3
        for c in C: s.bas.append(nn.Sequential(Bloc(a, c), Bloc(c, c))); a = c
        for c in reversed(C[:-1]): s.haut.append(nn.Sequential(Bloc(a + c, c), Bloc(c, c))); a = c
        s.fin = nn.Conv2d(a, 6, 1)
    def forward(s, x):
        P = []
        for i, b in enumerate(s.bas):
            x = b(x if i == 0 else F.avg_pool2d(x, 2)); P.append(x)
        for i, b in enumerate(s.haut): x = b(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), P[-2 - i]], 1))
        return s.fin(x)
def enr(a): return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
def _ero(x): return -F.max_pool2d(-x, 3, 1, 1)
def _ouv(x): return F.max_pool2d(_ero(x), 3, 1, 1)
def squelette_doux(x, n=6):
    sk = F.relu(x - _ouv(x))
    for _ in range(n):
        x = _ero(x); d = F.relu(x - _ouv(x)); sk = sk + F.relu(d - sk * d)
    return sk
def continuite(pred, vrai, poids):
    # pred, vrai, poids : (lot, H, W). Part du squelette prédit qui tombe sur un vrai trait, part du squelette vrai couverte par la prédiction.
    sp = squelette_doux(pred[:, None])[:, 0] * poids; sv = squelette_doux(vrai[:, None])[:, 0] * poids; a = (sp * vrai).sum() / (sp.sum() + 1.0); b = (sv * pred).sum() / (sv.sum() + 1.0); return 1 - 2 * a * b / (a + b + 1e-6)
def perte(o, c):
    lg, m = o[:, 0], c[:, 3]; cl = c[:, 0]; syn = 1 - c[:, 5]; w = (1 + 7 * c[:, 4]) * syn                          # poids 4 dans les zones cassées ; ligne et phase : coupes SYNTHÉTIQUES seulement
    b = (F.binary_cross_entropy_with_logits(lg, cl, pos_weight=torch.tensor(4.0, device=o.device), reduction="none") * w).sum() / (w.sum() + 1); p = torch.sigmoid(lg)
    dice = 1 - (2 * (p * cl * w).sum() + 1) / ((p * p * w).sum() + (cl * cl * w).sum() + 1); ph = ((o[:, 1] - c[:, 1]).abs() + (o[:, 2] - c[:, 2]).abs()) * m * w; ph = ph.sum() / (2 * (m * w).sum() + 1)
    # PENTE du numéro de tour (tours par pixel x 10) : enseignée partout où la vérité existe, y compris sur les VRAIES coupes (zone valide = entre deux tours consécutifs du déroulé publié)
    wp = m * (1 + 7 * c[:, 4]); vp = torch.atan2(c[:, 1], c[:, 2]); gx = enr(vp[:, :, 1:] - vp[:, :, :-1]) * (10 / (2 * math.pi)); gy = enr(vp[:, 1:, :] - vp[:, :-1, :]) * (10 / (2 * math.pi))
    mx = wp[:, :, 1:] * (m[:, :, :-1] > 0) * (gx.abs() < 4.5); my = wp[:, 1:, :] * (m[:, :-1, :] > 0) * (gy.abs() < 4.5)
    pe = (((o[:, 3][:, :, 1:] - gx).abs() * mx).sum() + ((o[:, 4][:, 1:, :] - gy).abs() * my).sum()) / (mx.sum() + my.sum() + 1)
    # DENSITÉ : log(tours par pixel x 10), vérité = longueur de la pente vraie (différences centrées de la phase vraie) ; enseignée aussi sur les vraies coupes
    cgx = (enr(vp[:, 1:-1, 2:] - vp[:, 1:-1, 1:-1]) + enr(vp[:, 1:-1, 1:-1] - vp[:, 1:-1, :-2])) * (5 / (2 * math.pi)); cgy = (enr(vp[:, 2:, 1:-1] - vp[:, 1:-1, 1:-1]) + enr(vp[:, 1:-1, 1:-1] - vp[:, :-2, 1:-1])) * (5 / (2 * math.pi))
    dv = torch.sqrt(cgx ** 2 + cgy ** 2); md = wp[:, 1:-1, 1:-1] * m[:, 1:-1, 2:] * m[:, 1:-1, :-2] * m[:, 2:, 1:-1] * m[:, :-2, 1:-1] * (dv > 0.15) * (dv < 4.5)
    de = ((o[:, 5][:, 1:-1, 1:-1] - torch.log(dv.clamp(min=0.15))).abs() * md).sum() / (md.sum() + 1)
    # NORME : (sin, cos) de longueur 1 (un trait qui s'arrête ou deux traits qui fusionnent = longueur 0)
    no = (((torch.sqrt(o[:, 1] ** 2 + o[:, 2] ** 2 + 1e-6) - 1) ** 2) * m).sum() / (m.sum() + 1)
    # COHÉRENCE (sans vérité, donc aussi sur le réel) : la phase prédite doit avancer d'un pixel au suivant comme le dit la pente prédite -> une feuille par tour
    pp = torch.atan2(o[:, 1], o[:, 2]); cx_ = enr(pp[:, :, 1:] - pp[:, :, :-1]) * (10 / (2 * math.pi)); cy_ = enr(pp[:, 1:, :] - pp[:, :-1, :]) * (10 / (2 * math.pi))
    kx = m[:, :, 1:] * m[:, :, :-1]; ky = m[:, 1:, :] * m[:, :-1, :]
    coh = (((cx_ - o[:, 3][:, :, 1:].detach()).abs().clamp(max=3) * kx).sum() + ((cy_ - o[:, 4][:, 1:, :].detach()).abs().clamp(max=3) * ky).sum()) / (kx.sum() + ky.sum() + 1)
    # CONTINUITÉ des traits (coupes synthétiques) : sur la carte de lignes, et sur le trait tiré de la phase (cos proche de 1, celui qu'on dessine)
    vl = (cl > 0.3).float(); pc = 1 + 3 * c[:, 4]; lphi = torch.sigmoid(10 * (o[:, 2] / torch.sqrt(o[:, 1] ** 2 + o[:, 2] ** 2 + 1e-6) - 0.75)) * m
    ct = continuite(p * syn, vl * syn, pc) + continuite(lphi * syn, F.max_pool2d(vl[:, None], 3, 1, 1)[:, 0] * syn, pc)
    return b + dice + 2 * ph + 2 * pe + 1.5 * no + 1.0 * coh + 2 * de + 1.5 * ct, (b.item(), dice.item(), ph.item(), pe.item(), no.item(), coh.item(), de.item(), ct.item())
# ---------------------------------------------------------------- contrôle sur une coupe entière
@torch.no_grad()
def tracer(net, img, centre, pas=384):
    H, W = img.shape; x = torch.from_numpy(img.astype(np.float32) / 255.0); R_ = torch.from_numpy(np.hypot(np.arange(W, dtype=np.float32)[None, :] + 0.5 - centre[0], np.arange(H, dtype=np.float32)[:, None] + 0.5 - centre[1]))
    def entree(y, x_):
        r = R_[y:y + T, x_:x_ + T]; return torch.stack([x[y:y + T, x_:x_ + T], (r - r.mean()) / 256.0, r / 3000.0])
    pass; acc = torch.zeros(6, H, W); poids = torch.zeros(H, W); w = torch.hann_window(T, periodic=False).clamp(min=0.02); w2 = w[:, None] * w[None]
    ys = list(range(0, max(H - T, 0) + 1, pas)); xs = list(range(0, max(W - T, 0) + 1, pas))
    if ys[-1] != H - T: ys.append(H - T)
    if xs[-1] != W - T: xs.append(W - T)
    P = [(y, x_) for y in ys for x_ in xs if img[y:y + T, x_:x_ + T].max() > 12]
    for i in range(0, len(P), 24):
        lot = P[i:i + 24]; b = torch.stack([entree(y, x_) for y, x_ in lot]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16): o = net(b).float().cpu()
        for (y, x_), oo in zip(lot, o): acc[:, y:y + T, x_:x_ + T] += oo * w2; poids[y:y + T, x_:x_ + T] += w2
    acc /= poids.clamp(min=1e-6); return torch.sigmoid(acc[0]).numpy(), np.arctan2(acc[1].numpy(), acc[2].numpy())
def cretes(p, seuil=0.3):
    from skimage.morphology import skeletonize                                                                 # 26/11 : les maxima locaux donnaient des POINTILLÉS (note faussée) -> squelette continu de la carte seuillée
    return skeletonize(p >= seuil)
def lignes_phase(ph, dedans):
    s_ = np.sin(ph); c_ = np.cos(ph) > 0.2; L = np.zeros(ph.shape, bool)
    L[:, :-1] |= (s_[:, :-1] * s_[:, 1:] <= 0) & c_[:, :-1] & c_[:, 1:]; L[:-1, :] |= (s_[:-1, :] * s_[1:, :] <= 0) & c_[:-1, :] & c_[1:, :]; return L & dedans
def interieur(img):
    m = (cv2.blur(img, (61, 61)) > 22).astype(np.uint8); return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8)) > 0
def bouts_par_mpx(L, dedans):
    from skimage.morphology import skeletonize
    S = skeletonize(L); nv = cv2.filter2D(S.astype(np.uint8), -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT); d = cv2.erode(dedans.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    return round(float((S & (nv == 2) & d).sum()) / max(float(d.sum()) / 1e6, 1e-6), 1)
def accord(pred, vrai, zone, tol):
    dv = cv2.distanceTransform((~vrai).astype(np.uint8), cv2.DIST_L2, 3); dp = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    pr = float((dv[pred & zone] <= tol).mean()) if (pred & zone).any() else 0.0; ra = float((dp[vrai & zone] <= tol).mean()) if (vrai & zone).any() else 0.0
    return pr, ra, (2 * pr * ra / (pr + ra) if pr + ra else 0.0)
def planche(nom, img, pred, vrai, pas_):
    H, W = img.shape; s = 1100; y, x = max(0, H // 2 - s // 2 - H // 8), max(0, W // 2 - s // 2 + W // 8); a = cv2.cvtColor(img[y:y + s, x:x + s], cv2.COLOR_GRAY2BGR); b, c = a.copy(), a.copy()
    b[cv2.dilate(pred[y:y + s, x:x + s].astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (0, 228, 240); c[cv2.dilate(vrai[y:y + s, x:x + s].astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (230, 159, 0)
    for im, t in ((a, f"{nom} - CT slice"), (b, f"model trace (yellow), step {pas_}"), (c, "ground truth / published trace (blue)")): cv2.putText(im, t, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(f"{D}/planches/{nom}_pas{pas_:06d}.jpg", np.hstack([a, b, c]), [1, 86])
def controle(net, pas_):
    net.eval(); R = dict(pas=pas_); F1 = []
    for n in sorted(TEMOINS):
        p = f"{D}/prep/{n}.npy"
        if not os.path.exists(p): continue
        a = np.load(p, mmap_mode="r"); img = np.array(a[0]); lig = np.array(a[1]) > 0; ded = np.array(a[3]) > 0
        try: ctr = json.load(open(p[:-4] + ".json"))["centre"]
        except Exception: continue
        pr, ph = tracer(net, img, ctr); cr = lignes_phase(ph, ded); P_, R_, f = accord(cr, lig, ded, 2.0); cas = cv2.dilate((np.array(a[4]) > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0; rv = accord(cr, lig, ded & cas, 2.0)
        vph = np.array(a[2]).astype(np.float32) * (2 * np.pi / 256); eph = float(np.abs(np.angle(np.exp(1j * (ph - vph))))[ded].mean() / np.pi * 180)
        R[n] = dict(precision=round(P_, 3), rappel=round(R_, 3), f1=round(f, 3), phase_deg=round(eph, 1), rappel_dans_le_vide=round(rv[1], 3), f1_carte=round(accord(cretes(pr), lig, ded, 2.0)[2], 3), bouts_par_mpx=bouts_par_mpx(cr, ded)); F1.append(f); planche(n, img, cr, lig, pas_)
    if os.path.exists(D + "/temoins/paris4_z9024.png"):
        img = cv2.imread(D + "/temoins/paris4_z9024.png", 0); tour = np.load(D + "/temoins/paris4_z9024.npz")["tour"]; vrai = tour > 0; zone = cv2.dilate(vrai.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        pr, ph = tracer(net, img, json.load(open(D + "/temoins/paris4_centre.json"))["centre"]); ded_p4 = interieur(img); cr = lignes_phase(ph, ded_p4); P_, R_, f = accord(cr, vrai, zone, 3.0); R["paris4"] = dict(bouts_par_mpx=bouts_par_mpx(cr, ded_p4), precision=round(P_, 3), rappel=round(R_, 3), f1=round(f, 3), f1_carte=round(accord(cretes(pr), vrai, zone, 3.0)[2], 3), lignes_par_trace=round(float(cr[zone].sum()) / max(float(vrai.sum()), 1), 2)); planche("paris4", img, cr, vrai, pas_)
        np.save(D + "/planches/paris4_proba_dernier.npy", (pr * 255).astype(np.uint8))
    R["f1_synth"] = round(float(np.mean(F1)), 3) if F1 else None; open(D + "/eval.jsonl", "a").write(json.dumps(R) + "\n"); net.train(); return R
# ---------------------------------------------------------------- boucle
net = Traceur().to(dev).to(memory_format=torch.channels_last); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); gs = torch.amp.GradScaler("cuda"); pas = 0; meilleur = -1.0
if os.path.exists(D + "/dernier.pt"):
    s = torch.load(D + "/dernier.pt", map_location=dev); e = s["net"]; neuf = False
    for k in ("bas.0.0.c1.weight", "bas.0.0.sk.weight"):
        if k in e and e[k].shape[1] == 1:                                                                          # anciens poids à 1 entrée (image seule) : les 2 entrées de distance au centre partent de zéro
            w3 = torch.zeros(e[k].shape[0], 3, *e[k].shape[2:], device=e[k].device); w3[:, :1] = e[k]; e[k] = w3; neuf = True
    if e["fin.weight"].shape[0] == 5:                                                                              # anciens poids à 5 sorties : la sortie « densité » part de zéro
        w6 = torch.zeros(6, *e["fin.weight"].shape[1:], device=e["fin.weight"].device); w6[:5] = e["fin.weight"]; b6 = torch.zeros(6, device=e["fin.bias"].device); b6[:5] = e["fin.bias"]; e["fin.weight"], e["fin.bias"] = w6, b6; neuf = True
    if e["fin.weight"].shape[0] == 3:                                                                              # anciens poids à 3 sorties : on garde tout, les 2 sorties de pente partent de zéro
        w5 = torch.zeros(5, *e["fin.weight"].shape[1:], device=e["fin.weight"].device); w5[:3] = e["fin.weight"]; b5 = torch.zeros(5, device=e["fin.bias"].device); b5[:3] = e["fin.bias"]; e["fin.weight"], e["fin.bias"] = w5, b5; net.load_state_dict(e); note("poids à 3 sorties repris, 2 sorties de pente ajoutées (optimiseur remis à neuf)")
    else:
        net.load_state_dict(e)
        if neuf: note("poids repris, 2 entrees de distance au centre ajoutees (optimiseur remis a neuf)")
        elif "opt" in s: opt.load_state_dict(s["opt"])
    pas = s["pas"]; meilleur = s.get("meilleur", -1.0); note(f"reprise au pas {pas}")
note(f"réseau : {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M de paramètres | lot {LOT} | témoins {sorted(TEMOINS)} + Paris 4")
dl = torch.utils.data.DataLoader(Coupes(), batch_size=LOT, num_workers=int(os.environ.get("OUVRIERS", 8)), pin_memory=True, prefetch_factor=4); t0 = time.time(); moy = np.zeros(8); k = 0
for x, c in dl:
    if pas >= PAS_MAX or os.path.exists(D + "/STOP_ENTR"): break
    lr = 3e-4 * min(1.0, (pas + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(pas / PAS_MAX, 1.0))))
    for g in opt.param_groups: g["lr"] = lr
    x, c = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last), c.to(dev, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16): o = net(x)
    L, d = perte(o.float(), c)
    if not torch.isfinite(L): note(f"pas {pas} : perte non finie, lot ignoré"); continue
    opt.zero_grad(set_to_none=True); gs.scale(L).backward(); gs.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); gs.step(opt); gs.update(); pas += 1; moy += d; k += 1
    if pas % 100 == 0:
        nb = len([p for p in glob.glob(D + "/prep/c*.npy") if os.path.basename(p)[:-4] not in TEMOINS]); nr = len(glob.glob(D + "/prep/r*.npy")); note(f"pas {pas} | reelles {nr} | ligne {moy[0] / k:.4f} dice {moy[1] / k:.4f} phase {moy[2] / k:.4f} pente {moy[3] / k:.4f} norme {moy[4] / k:.4f} coherence {moy[5] / k:.4f} densite {moy[6] / k:.4f} continuite {moy[7] / k:.4f} | {k / (time.time() - t0):.1f} pas/s | {nb} coupes"); moy[:] = 0; k = 0; t0 = time.time()
    if pas % EVAL == 0 or pas in (300, 700):
        R = controle(net, pas); p4 = R.get("paris4", {}).get("f1", -1.0); torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), pas=pas, meilleur=max(meilleur, p4)), D + "/dernier.pt")
        if p4 > meilleur: meilleur = p4; torch.save(dict(net=net.state_dict(), pas=pas, eval=R), D + "/meilleur.pt")
        note("CONTRÔLE " + json.dumps(R, ensure_ascii=False)); t0 = time.time(); k = 0; moy[:] = 0
note(f"ENTRAÎNEMENT ARRÊTÉ au pas {pas}")
