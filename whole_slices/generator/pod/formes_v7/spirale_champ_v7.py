"""24/11 · COUPE ENTIÈRE v7 = v6 + PÉTRISSAGE (bloc « v7 » après les fissures ; paramètre PRM["petrissage"] = {force, echelles, libre_rel, pas_px}).
24/11 · COUPE ENTIÈRE v6 (copie de spirale_champ_v5.py, NON modifié : un entraînement tourne avec formes_v5b).
Même principe : UNE feuille ininterrompue = courbes de niveau de F = u − φ/2π (champ continu W), toutes les déformations sont des difféomorphismes
(jacobien > 0 contrôlé partout), numéro de tour exact. Nouveautés v6 (écarts mesurés au même filtre que les vraies coupes, famille écrasée) :
  1. MASSIFS ÉCRASÉS : taches irrégulières dans (u, φ) couvrant une part de surface donnée (réel 5–29 %), espacement plafonné à 3,0–4,0 px, ondulations amorties dedans ;
  2. CHEVRONS (0814) : dans les massifs, bandes en zigzag = cisaillement pur le long de la normale (feuilles du paquet parallèles, jacobien = 1 + f·g' > 0) ;
  3. FROISSEMENTS : petits plis couchés serrés (quelques tours), même garde-fou que les plis (espacement ≥ 2 px) ;
     TOURBILLONS : rotation α(ρ) décroissante autour d'un point (feuilles tordues en S, paquets emmêlés de 0814), jacobien = 1 exactement ;
  4. ONDULATIONS à échelles libres « ond6 » [[σ, amplitude], …] : 24 px fort remplacé par 12–14 px (bande 128–512 px 3–4× trop forte en v5) ;
  5. RASTER par passages à zéro de W (lignes, sous-feuilles) : plus de squelette de bande seuillée, plus de pointillés quand deux niveaux sont proches.
Reste de v5 inchangé (espacement tour par tour, lentilles, côtés aplatis, bandes, cœur, plis, fissures).
    python spirale_champ_v6.py <parametres.json>     (voir tirer_parametres_v6.py)"""
import sys, os, time, json, numpy as np, cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
PRM = json.load(open(sys.argv[1], encoding='utf-8'))
NOM, DMM, SEED = PRM['nom'], PRM['diametre_mm'], PRM['graine']
UM = 9.4; O = os.path.dirname(os.path.abspath(__file__)) + '/sorties'; os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(SEED); t0 = time.time(); T = {}
R = DMM * 1000 / UM / 2
E = PRM['esp']; NPH = 1024; DU = 8                                     # échantillonnage du champ d'espacement : 1024 angles, 8 pas par tour
# ---------- 1. espacement tour par tour e(u, φ) ----------
emoy = E['touche_px'] * E['part_paquets'] + E['vide_med_px'] * np.exp(E['vide_sigma'] ** 2 / 2) * (1 - E['part_paquets'])
# v6 : les massifs (et bandes / côtés de la famille écrasée) tassent les tours -> sans correction la coupe sort plus petite que demandé (20 mm -> 14 mm à l'essai) ;
# on prévoit plus de tours pour garder le diamètre : espacement moyen effectif estimé avec la part de massif (≈ 3,2 px) et le tassement des zones (≈ 15 %).
_pm = PRM.get('massifs', {}).get('part', 0.0); emoy_eff = emoy * (1 - _pm) + (float(np.mean(PRM['massifs']['esp_px'])) if _pm > 0 else 3.2) * _pm
if _pm > 0: emoy_eff *= 0.85
N = int(R / emoy_eff) + 1; NU = (N + 3) * DU
uu = (np.arange(NU) / DU)[:, None]; ph = (np.arange(NPH) / NPH * 2 * np.pi - np.pi)[None, :]
def champ_polaire(cor_tours, cor_arc):
    """champ aléatoire lisse (écart-type 1) sur (u, φ), corrélé sur cor_tours tours et sur cor_arc px le long de la feuille (périodique en φ)"""
    g = rng.normal(0, 1, (NU, NPH)).astype(np.float32); out = np.zeros_like(g); refs = [2, 8, 25, 60, 150]
    for i, ur in enumerate(refs):
        sp = max(1.0, cor_arc / max(ur * emoy, 1.0) / (2 * np.pi / NPH))
        f = ndi.gaussian_filter(g, (cor_tours * DU, min(sp, NPH / 3)), mode=('nearest', 'wrap')); f /= f.std() + 1e-6
        lo = refs[i - 1] if i else -1; hi = refs[i + 1] if i + 1 < len(refs) else 1e9
        w = np.clip(np.minimum((uu - lo) / (ur - lo), (hi - uu) / (hi - ur)), 0, 1) if 0 < i < len(refs) - 1 else (np.clip((hi - uu) / (hi - ur), 0, 1) if i == 0 else np.clip((uu - lo) / (ur - lo), 0, 1))
        out += f * w
    return out
zb = champ_polaire(E['paquet_tours'], E['paquet_arc_px']); zg = champ_polaire(E['vide_tours'], E['vide_arc_px'])
q = np.quantile(zb, 1 - E['part_paquets']); pb = 1 / (1 + np.exp(-(zb - q) / 0.12))
e = pb * E['touche_px'] * np.exp(0.15 * zg) + (1 - pb) * E['vide_med_px'] * np.exp(E['vide_sigma'] * zg)
e = np.maximum(e, E['touche_px'] * 0.75)
e = e * (1 + E.get('tendance', 0.0) * (0.5 - np.clip(uu / N, 0, 1)))          # tours extérieurs un peu plus serrés (mesuré : remplissage qui monte vers le bord)
# FEUILLES QUI S'ÉCARTENT ET SE REJOIGNENT (lentilles de vide, paquets qui se font et se défont le long de la feuille, mesuré sur les zooms réels) :
# chaque tour reçoit un petit déplacement radial propre G(u, φ), corrélé sur peu de tours et sur 40–150 px le long de la feuille ;
# on l'ajoute comme DIFFÉRENCE entre tours voisins (e += ∂G/∂u) : le décalage reste borné (pas de marche au hasard qui s'accumule vers l'extérieur).
if E.get('lentilles_px', 0) > 0:
    Gl = champ_polaire(E.get('lentilles_tours', 0.6), E.get('lentilles_arc_px', 80)) * E['lentilles_px']
    e = e + np.diff(Gl, axis=0, prepend=Gl[:1]) * DU
# v6 · DÉRIVE BORNÉE : r(u, φ) = Σ e ; les variations d'espacement corrélées le long de la feuille (150–500 px) S'ACCUMULENT tour après tour (marche au hasard)
# -> grandes ondulations de 128–512 px (mesuré : 1 095 sans aucune ondulation ni pli, réel 260–650). On borne l'écart D entre le rayon cumulé et sa version
# lissée le long de la feuille (σ ≈ 'derive_arc_px') : D' = Dmax·tanh(D/Dmax). e' = (1 − g')·ē + g'·e (g' = pente de tanh) : mélange convexe de e et de
# l'espacement lissé, donc toujours > 0 (ordre des tours gardé) ; les paquets et vides locaux restent, seule la dérive à grande échelle est bornée.
DMX = E.get('derive_max_px', 0)
if DMX > 0:
    Rc = np.cumsum(e, 0) / DU; Rm_ = Rc.mean(1) + 1.0; Rs = np.empty_like(Rc); Es = np.empty_like(e)
    for i_ in range(NU):
        sg = float(np.clip(E.get('derive_arc_px', 300) / (Rm_[i_] * 2 * np.pi / NPH), 1, NPH / 4))
        Rs[i_] = ndi.gaussian_filter1d(Rc[i_], sg, mode='wrap'); Es[i_] = ndi.gaussian_filter1d(e[i_], sg, mode='wrap')
    Dv = Rc - Rs; gp = 1 / np.cosh(np.clip(Dv / DMX, -20, 20)) ** 2
    e = (1 - gp) * Es + gp * e
    del Rc, Rs, Es, Dv, gp
T['espacement'] = time.time() - t0
# ---------- 2. écrasements non uniformes ----------
def fenetre_u(a, b, bord):
    return np.clip((uu - a) / bord + 0.5, 0, 1) * np.clip((b - uu) / bord + 0.5, 0, 1)
def fenetre_phi(c, demi):
    d = np.angle(np.exp(1j * (ph - c))); x = np.clip(np.abs(d) / (1.6 * demi), 0, 1); return np.cos(np.pi / 2 * x) ** 2     # cloche douce : le bord aplati se raccorde sans marche
zones = []
for z in PRM['ecrasements']:
    if z['type'] == 'cote':
        continue                                                            # les côtés aplatis sont traités après le cœur (il faut le rayon du dernier tour)
    a, b = z['u_rel'][0] * N, z['u_rel'][1] * N; wz = fenetre_u(a, b, max(2.0, 0.08 * (b - a))) * fenetre_phi(np.radians(z['angle_deg']), np.radians(z['demi_secteur_deg']))
    if z.get('oppose'): wz = np.maximum(wz, fenetre_u(a, b, max(2.0, 0.08 * (b - a))) * fenetre_phi(np.radians(z['angle_deg'] + 180), np.radians(z['demi_secteur_deg'])))
    # irrégularité du bord de la zone le long de la feuille (les vraies zones ne sont pas des secteurs nets)
    wz = wz * np.clip(1 + 0.5 * champ_polaire(3, 200), 0, 1.2).clip(0, 1)
    e = e * (1 - wz) + np.minimum(e, z['esp_px']) * wz; zones.append(z)
# v6 · MASSIFS ÉCRASÉS (mesuré sur 0306B / 0814 / 0139 z12448 : 5–29 % des fenêtres de 64 px remplies ≥ 80 %, zones de 50–115 tours de profondeur,
# formes irrégulières, pas des secteurs) : taches d'un champ lisse dans (u, φ), seuillées pour couvrir la part de SURFACE voulue (pondérée par u ≈ r),
# bord doux ; dedans, l'espacement est plafonné à 3,0–4,0 px (feuilles qui se touchent, plus de vide). Écrasement LOCAL : le reste garde son espacement.
MS = PRM.get('massifs', {}); Wm = np.zeros_like(e)
if MS.get('part', 0) > 0:
    zm = champ_polaire(MS.get('cor_tours', 10), MS.get('cor_arc_px', 900))
    poids = np.clip(uu, 0.5, None) * (uu > MS.get('u_min_rel', 0.1) * N) * (uu < MS.get('u_max_rel', 0.93) * N) * np.ones_like(zm)                     # surface ≈ ∝ r dr dφ ≈ ∝ u du dφ
    qs = np.sort(zm.ravel()); cw = np.cumsum(poids.ravel()[np.argsort(zm.ravel())]); cw /= cw[-1]
    seuil_m = qs[np.searchsorted(cw, 1 - MS['part'])]
    Wm = 1 / (1 + np.exp(-(zm - seuil_m) / 0.08)) * np.clip((uu - MS.get('u_min_rel', 0.1) * N) / 3, 0, 1) * np.clip((MS.get('u_max_rel', 0.93) * N - uu) / 3, 0, 1)   # massifs DANS la coupe (le bord est déjà tassé par les côtés)
    Wm = np.clip((ndi.gaussian_filter1d(Wm, 2 * DU, axis=0) - 0.15) / 0.6, 0, 1)       # massif plein : pas de fines lacunes de bord entre paquets (vu sur v6_ec1 : liserés sombres)
    em = MS['esp_px'][0] + (MS['esp_px'][1] - MS['esp_px'][0]) * (0.5 + 0.5 * np.tanh(champ_polaire(2, 200)))
    e = e * (1 - Wm) + np.minimum(e, em) * Wm
# ---------- 3. cœur : trou et lentille ----------
C = PRM['coeur']; phc = np.radians(C['angle_deg'])
if C['lentille'] > 0:
    wc = np.clip(1 - uu / C['tours'], 0, 1) ** 1.5; g = np.abs(np.cos(ph - phc)) ** 1.5
    e = e * (1 - wc * C['lentille'] * (1 - g))                       # premiers tours : espacement réduit dans la direction courte -> lentille
e = np.maximum(e, PRM.get('plancher_px', 2.6))                       # v7 : plancher réglable (5 px ≈ 47 µm)
r0 = C['trou_px'] * (1 + 0.35 * np.cos(ph - phc) * C['lentille'] + 0.25 * ndi.gaussian_filter1d(rng.normal(0, 1, NPH), 40, mode='wrap') * 4)
r0 = np.maximum(r0, 1.0)
# CÔTÉS APLATIS (mesuré : contours en trapèze / triangle, bords droits, tours extérieurs tassés contre le bord droit) :
# le dernier tour doit tenir derrière une corde à distance d du centre ; l'excédent de rayon est absorbé en tassant les tours extérieurs du secteur.
iN = min(NU - 1, N * DU)
for z in PRM['ecrasements']:
    if z['type'] != 'cote': continue
    rN = (r0 + e[:iN].sum(0) / DU).ravel(); p0 = np.radians(z['angle_deg']); dphi = np.angle(np.exp(1j * (ph[0] - p0)))
    dd = np.median(rN[np.abs(dphi) < 0.2]) * np.cos(np.radians(z['demi_secteur_deg']))
    corde = np.where(np.abs(dphi) < np.pi / 2 * 0.98, dd / np.maximum(np.cos(dphi), 1e-3), 1e9)
    exc = ndi.gaussian_filter1d(np.clip(rN - corde, 0, None), 6, mode='wrap')       # coins légèrement arrondis
    u1 = z['u_rel'][0] * N; Wb = np.clip((uu - u1) / 4.0, 0, 1) * (uu < N + 1)
    I = (e * Wb).sum(0) / DU; c = np.clip(exc / np.maximum(I, 1e-6), 0, 1)
    e = np.maximum(e * (1 - c[None, :] * Wb), np.minimum(e, z['esp_px'])); zones.append(dict(z, corde_px=round(float(dd))))
rU = r0 + np.concatenate([np.zeros((1, NPH)), np.cumsum(e[:-1], 0) / DU], 0)          # rayon final de (u, φ), strictement croissant en u
if PRM.get('taille_exacte'):                                             # v7.1 : avec les écarts élargis le nombre de tours prévu donnait une coupe 1,3–1,8× trop grande ;
    N = max(8, int(np.searchsorted(rU.mean(1), R) / DU))                  # on garde seulement les tours qui tiennent dans le rayon demandé
T['ecrasements'] = time.time() - t0
# ---------- 4. table inverse u(r, φ) ----------
rmax = float(rU[min(NU - 1, (N + 1) * DU)].max()) * 1.02 + 40; dr = 0.5; nr = int(rmax / dr) + 2; rg = np.arange(nr) * dr
tab = np.empty((NPH + 1, nr), np.float32)
for j in range(NPH): tab[j] = np.interp(rg, rU[:, j], uu[:, 0], left=-1.0, right=float(uu[-1, 0]))
tab[NPH] = tab[0]
# ---------- 5. grille finale et déformations (toutes en « tirage arrière » final -> base) ----------
ov = PRM['ovale']; k = ov['rapport']; al = np.radians(ov['angle_deg'])
MARGE7 = 2.5 * PRM.get('petrissage', {}).get('force', 0) + 0.7 * max([abs(q_['pente']) * q_['demi_largeur_rel'] for q_ in PRM.get('plis_francs', [])] + [0.0])                 # v7 : canevas élargi, le rouleau pétri se déforme librement (bord compris)
FA7 = float(np.sqrt(1 + 2 * PRM.get('amande', {}).get('h_rel', 0)))         # v7.3 : un rouleau aplati « en amande » est plus long dans l'axe des épingles
H = int(2 * rmax / np.sqrt(k) * (1.04 + MARGE7) * FA7 + 64); H += H % 2; W = H
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); xx -= W / 2; yy -= H / 2
def champ(sig):
    f = max(1, int(sig // 6)); g = rng.normal(0, 1, (H // f + 2, W // f + 2)).astype(np.float32); g = ndi.gaussian_filter(g, sig / f); g /= g.std() + 1e-6
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
# 5a. ondulations (partagées par les feuilles voisines)
dX = np.zeros((H, W), np.float32); dY = np.zeros((H, W), np.float32)
# v6 : échelles libres [[σ px, amplitude], …] (ond6) ; v5 avait 24 px fort -> bande 128–512 px 3–4× trop forte (pic de courbure à λ ≈ 4,4 σ ≈ 107 px).
ECH = PRM['ond6'] if 'ond6' in PRM else list(zip((24, 90, 300) if len(PRM['ond']) == 3 else (10, 24, 90, 300), PRM['ond']))
for sig, amp in ECH:
    if amp > 0: dX += amp * sig * champ(sig); dY += amp * sig * champ(sig)
# v6 : coordonnées de base approximatives de chaque pixel final (sans ondulation ni pli) -> poids « massif » sur la grille finale
def vers_base_approx(Xf, Yf):
    xr_, yr_ = np.cos(al) * Xf + np.sin(al) * Yf, -np.sin(al) * Xf + np.cos(al) * Yf; Xb_, Yb_ = xr_ * np.sqrt(k), yr_ / np.sqrt(k)
    rb_, pb2 = np.hypot(Xb_, Yb_), np.arctan2(Yb_, Xb_); my2 = ((pb2 + np.pi) / (2 * np.pi) * NPH).astype(np.float32)
    U_ = cv2.remap(tab, (rb_ / dr).astype(np.float32), my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(uu[-1, 0]))
    return U_, my2
Ua, Pa = vers_base_approx(xx, yy)
Wf = cv2.remap(Wm.astype(np.float32), Pa, (np.clip(Ua, 0, None) * DU).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
Wf = cv2.GaussianBlur(Wf, (0, 0), 25); del Ua, Pa
amort = 1 - MS.get('amorti_ond', 0.0) * np.clip(Wf, 0, 1)                    # massifs : feuilles plus droites et parallèles (vrais massifs laminaires) ; le flot reste un difféomorphisme
dX *= amort; dY *= amort
# intégré en 6 petits pas (flot) : chaque pas est un difféomorphisme même là où la pente du déplacement dépasse 1 -> jacobien > 0 garanti
X, Y = xx.copy(), yy.copy(); KO = 14
for _ in range(KO):
    mx_, my_ = X + W / 2, Y + H / 2
    X, Y = X + cv2.remap(dX, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / KO, Y + cv2.remap(dY, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / KO
del dX, dY
T['ondulations'] = time.time() - t0
GPU = os.environ.get('GPU', '0') == '1'
if GPU: import torch
EMIN = float(PRM.get('plis_esp_min', 2.0))                               # v7.1 : écart minimal toléré dans un pli / chevron / tourbillon (vrais tracés : presque rien sous 7 px)
def avant(u, p):
    """position finale approximative (sans ondulation ni pli) du point de base (u, φ) : rayon r(u, φ), puis ovale"""
    j = int(round((p + np.pi) / (2 * np.pi) * NPH)) % NPH; r = np.interp(u, uu[:, 0], rU[:, j]); x, y = r * np.cos(p), r * np.sin(p)
    xr, yr = x / np.sqrt(k), y * np.sqrt(k); return np.cos(al) * xr - np.sin(al) * yr, np.sin(al) * xr + np.cos(al) * yr
# 5b. plis en épingle : plusieurs tours repliés ENSEMBLE. Bosse à plateau (les tours du paquet bougent d'un bloc, gardent leur espacement ;
#     les tours au-dessus absorbent la bosse en douceur) puis cisaillement le long des feuilles (renverse la bosse = pli couché, la feuille fait demi-tour).
#     Garde-fou : le pli est d'abord essayé sur un paquet droit ; on réduit hauteur et renversement tant que l'espacement minimal < 2,6 px ou le jacobien ≤ 0.
def pli_tirage(t, n, st, D, A, kap):
    """tirage arrière d'un pli : (t, n) final -> (t, n) de base. Aller : bosse à plateau de hauteur H = A·D (paquet |n| < D qui bouge d'un bloc),
    puis cisaillement le long des feuilles proportionnel à la hauteur (rampe q de −D à D + H) : le sommet de la bosse dépasse sa base = pli couché."""
    H = A * D; lam = max(1.5 * H, D); S = kap * st / max(H, 1e-6); c = H / 2; L = D + H / 2; sm = max(3.5 * st, 1.2 * S * L)
    g_ = GPU and t.size > 20000; xp = torch if g_ else np                # 24/11 : mêmes formules sur la carte graphique pour les grandes zones (vitesse)
    if g_: t = torch.from_numpy(np.ascontiguousarray(t, np.float32)).cuda(); n = torch.from_numpy(np.ascontiguousarray(n, np.float32)).cuda()
    w0 = float(np.tanh(D / lam)); m = lambda t_: xp.exp(-t_ ** 2 / (2 * sm ** 2)); qf = lambda n_: L * xp.tanh((n_ - c) / L) * xp.exp(-((n_ - c) / (1.5 * L)) ** 2)
    t1 = t; qn = S * qf(n)
    for _ in range(60): t1 = t - qn * m(t1)
    bb = xp.exp(-t1 ** 2 / (2 * st ** 2)); nb = n
    for _ in range(60): nb = n - H * bb * 0.5 * (xp.tanh((nb + D) / lam) - xp.tanh((nb - D) / lam)) / w0
    if g_: t1, nb = t1.cpu().numpy(), nb.cpu().numpy()
    return t1, nb, sm, 6 * L + 4 * lam
def pli_sur(st, D, A, kap, e_loc):
    for essai in range(10):
        _, _, sm, Lq = pli_tirage(np.zeros(1), np.zeros(1), st, D, A, kap)
        tt, nn = np.meshgrid(np.arange(-3 * sm, 3 * sm, 2.0), np.arange(-Lq / 3, Lq / 3, 2.0))      # essai au pas de 2 px (rapide)
        t1, nb, _, _ = pli_tirage(tt, nn, st, D, A, kap)
        gy_, gx_ = np.gradient(nb, 2.0); jt_y, jt_x = np.gradient(t1, 2.0); jac = jt_x * gy_ - jt_y * gx_
        esp_min = e_loc / np.hypot(gx_, gy_).max()
        if esp_min >= EMIN and jac.min() > 0.02: return A, kap, esp_min, essai
        A *= 0.85; kap *= 0.8
    return 0.0, 0.0, esp_min, essai
plis = []; occ = []                                                     # v6 : disques occupés (pas de chevauchement de déformations locales)
for P_ in PRM['plis']:
    up, pp = P_['u_rel'] * N, np.radians(P_['angle_deg'])
    # le pli se place là où les tours ont de la place (sinon le garde-fou l'écrase) : 8 positions voisines essayées, on garde la plus lâche
    cands = [(up, pp)] + [(float(np.clip(up + rng.normal(0, 0.06 * N), 0.2 * N, 0.93 * N)), pp + rng.normal(0, 0.5)) for _ in range(7)]
    def e_ici(u_, p_):
        j_ = int(round((np.angle(np.exp(1j * p_)) + np.pi) / (2 * np.pi) * NPH)) % NPH; i_ = int(u_ * DU); return float(np.percentile(e[max(0, i_ - 3 * DU):i_ + 3 * DU, j_], 30))
    up, pp = max(cands, key=lambda c_: e_ici(*c_)); pp = float(np.angle(np.exp(1j * pp)))
    xa, ya = avant(up, pp - 0.01); xb, yb = avant(up + 0.02 / (2 * np.pi), pp + 0.01); px_, py_ = avant(up, pp)
    et = np.array([xb - xa, yb - ya]); et /= np.linalg.norm(et) + 1e-9; en = np.array([-et[1], et[0]])
    if en @ np.array([px_, py_]) < 0: en = -en
    st, D = P_['largeur_px'], P_['profondeur_px']
    j = int(round((pp + np.pi) / (2 * np.pi) * NPH)) % NPH; iu = int(up * DU); e_loc = float(np.percentile(e[max(0, iu - 3 * DU):iu + 3 * DU, j], 30))
    A, kap, esp_min, essais = pli_sur(st, D, P_['hauteur'], P_['renversement'], e_loc)
    if A <= 0: continue
    _, _, sm, Lq = pli_tirage(np.zeros(1), np.zeros(1), st, D, A, kap)
    rb_ = np.hypot(4.5 * sm, Lq) + 20; cyb, cxb = int(py_ + H / 2), int(px_ + W / 2); occ.append((px_, py_, 0.6 * rb_))
    by0, by1, bx0, bx1 = max(0, int(cyb - rb_)), min(H, int(cyb + rb_)), max(0, int(cxb - rb_)), min(W, int(cxb + rb_))
    Xb, Yb = X[by0:by1, bx0:bx1], Y[by0:by1, bx0:bx1]                   # vues : modifiées sur place
    t = (Xb - px_) * et[0] + (Yb - py_) * et[1]; n = (Xb - px_) * en[0] + (Yb - py_) * en[1]
    zone = (np.abs(t) < 4.5 * sm) & (np.abs(n) < Lq)
    t1, nb, _, _ = pli_tirage(t[zone], n[zone], st, D, A, kap)
    Xb[zone] = px_ + t1 * et[0] + nb * en[0]; Yb[zone] = py_ + t1 * et[1] + nb * en[1]
    plis.append(dict(P_, hauteur_retenue=round(A, 2), renversement_retenu=round(kap, 2), esp_min_px=round(esp_min, 1), essais=essais, x=round(float(px_)), y=round(float(py_))))
T['plis'] = time.time() - t0
# v6 · DANS LES MASSIFS ÉCRASÉS : chevrons (bandes en zigzag, 0814) et petits froissements (plis couchés serrés, quelques tours).
# Position tirée DANS les massifs (poids ≥ 0,7, proba ∝ surface) ; espacement local = 10e centile (garde-fou au plus serré).
iM = np.argwhere((Wm > 0.3) & (uu < N - 1)); iT = np.argwhere((uu > 0.15 * N) & (uu < N - 1) & np.ones_like(Wm, bool))
def libre(u_, p_, ray):
    x_, y_ = avant(u_, p_); return all(np.hypot(x_ - a_, y_ - b_) > ray + r_ for a_, b_, r_ in occ)
def tirer_dans_massif(ou=None, cands=1, ray=0.0):
    """position (u, φ) tirée dans les massifs (ou partout : ou = iT), proba ∝ surface, hors des disques déjà occupés (déformations locales qui se chevauchent
    = compressions qui s'additionnent : 2 pixels à jacobien < 0 à l'essai t4) ; parmi « cands » essais libres on garde le plus lâche (10e centile)"""
    ou = iM if ou is None else ou
    if not len(ou): return None
    w_ = ou[:, 0].astype(float) + 1; best = None
    for _ in range(cands * 20):
        i_ = rng.choice(len(ou), p=w_ / w_.sum()); c_ = (ou[i_, 0] / DU, ou[i_, 1] / NPH * 2 * np.pi - np.pi)
        if not libre(*c_, ray): continue
        if best is None or e_min_ici(*c_) > e_min_ici(*best): best = c_
        cands -= 1
        if cands <= 0: break
    return best
def repere(up, pp):
    xa, ya = avant(up, pp - 0.01); xb, yb = avant(up + 0.02 / (2 * np.pi), pp + 0.01); px_, py_ = avant(up, pp)
    et = np.array([xb - xa, yb - ya]); et /= np.linalg.norm(et) + 1e-9; en = np.array([-et[1], et[0]])
    if en @ np.array([px_, py_]) < 0: en = -en
    return px_, py_, et, en
def e_min_ici(up, pp, pc=10):
    j_ = int(round((pp + np.pi) / (2 * np.pi) * NPH)) % NPH; i_ = int(up * DU); return float(np.percentile(e[max(0, i_ - 2 * DU):i_ + 2 * DU, j_], pc))
def appliquer_local(px_, py_, et, en, rt, rn, f_tirage):
    """compose le tirage arrière local (t, n) -> (t', n') dans le rectangle |t| < rt, |n| < rn autour de (px_, py_)"""
    rb2 = np.hypot(rt, rn) + 4; cyb, cxb = int(py_ + H / 2), int(px_ + W / 2)
    by0, by1, bx0, bx1 = max(0, int(cyb - rb2)), min(H, int(cyb + rb2)), max(0, int(cxb - rb2)), min(W, int(cxb + rb2))
    Xb, Yb = X[by0:by1, bx0:bx1], Y[by0:by1, bx0:bx1]
    t = (Xb - px_) * et[0] + (Yb - py_) * et[1]; n = (Xb - px_) * en[0] + (Yb - py_) * en[1]; zone = (np.abs(t) < rt) & (np.abs(n) < rn)
    t1, nb = f_tirage(t[zone], n[zone]); Xb[zone] = px_ + t1 * et[0] + nb * en[0]; Yb[zone] = py_ + t1 * et[1] + nb * en[1]
def chevron_tirage(t, n, A, lam, Lt, Ln, ph0, wob):
    """aller : n' = n + f(t)·h(t)·g(n), f = zigzag d'amplitude A (coins arrondis), h, g = plateaux doux. Cisaillement pur le long de la normale :
    les feuilles du paquet restent parallèles (aucun croisement) ; jacobien = 1 + f·h·g'(n) > 0 car A ≤ sn/3 (tassement de bord ≤ 1,2×). Inverse par point fixe (contractant)."""
    st_ = max(0.35 * lam, 10.0); sn = max(3.0 * A, 0.35 * Ln, 12.0)
    g_ = GPU and t.size > 20000; xp = torch if g_ else np; t0_ = t
    if g_: t = torch.from_numpy(np.ascontiguousarray(t, np.float32)).cuda(); n = torch.from_numpy(np.ascontiguousarray(n, np.float32)).cuda()
    th_ = 2 * np.pi * t / lam + ph0 + wob[0] * xp.sin(2 * np.pi * t / (2.7 * lam) + wob[1]) + wob[2] * xp.sin(2 * np.pi * t / (1.3 * lam) + wob[3])
    f = A * (2 / np.pi) * xp.arcsin(0.96 * xp.sin(th_)) / float(2 / np.pi * np.arcsin(0.96)) * 0.5 * (xp.tanh((t + Lt) / st_) - xp.tanh((t - Lt) / st_))
    g = lambda n_: 0.5 * (xp.tanh((n_ + Ln) / sn) - xp.tanh((n_ - Ln) / sn))
    nb = n
    for _ in range(40): nb = n - f * g(nb)
    if g_: nb = nb.cpu().numpy()
    return t0_, nb
def chevron_sur(A, lam, Lt, Ln, ph0, wob, e_loc):
    for essai in range(12):
        sn = max(3.0 * A, 0.35 * Ln, 12.0)
        tt, nn = np.meshgrid(np.arange(-Lt - lam, Lt + lam, 1.0), np.arange(-Ln - 2 * sn, Ln + 2 * sn, 1.0))
        _, nb = chevron_tirage(tt, nn, A, lam, Lt, Ln, ph0, wob); gy_, gx_ = np.gradient(nb, 1.0); gn_ = np.hypot(gx_, gy_)
        esp_min = e_loc / gn_.max()
        if esp_min >= EMIN and gy_.min() > 0.05: return A, esp_min, essai
        A *= 0.85
    return 0.0, esp_min, essai
chevrons = []; froiss = []
CV = PRM.get('chevrons', {})
for _ in range(CV.get('nombre', 0)):
    lam = float(rng.uniform(*CV['longueur_onde_px'])); pente = float(rng.uniform(*CV['pente'])); A = pente * lam / 4
    Lt = float(rng.uniform(*CV['long_px'])) / 2; Ln = float(rng.uniform(*CV['epais_px'])) / 2; ph0 = float(rng.uniform(0, 2 * np.pi))
    ray = 0.6 * np.hypot(Lt + lam / 2, Ln + max(3.0 * A, 0.35 * Ln, 12.0))      # rayon d'exclusion (les bords du chevron s'éteignent en douceur)
    pos = tirer_dans_massif(cands=CV.get('essais', 1), ray=ray)
    if pos is None: continue
    up, pp = pos; e_loc = e_min_ici(up, pp); px_, py_, et, en = repere(up, pp)
    wob = (float(rng.uniform(0, 1.2)), float(rng.uniform(0, 6.3)), float(rng.uniform(0, 0.8)), float(rng.uniform(0, 6.3)))
    A, esp_min, ess = chevron_sur(A, lam, Lt, Ln, ph0, wob, e_loc)
    if A < 1.0: continue
    sn = max(3.0 * A, 0.35 * Ln, 12.0)
    appliquer_local(px_, py_, et, en, Lt + 1.2 * lam + 30, Ln + 3 * sn, lambda t_, n_: chevron_tirage(t_, n_, A, lam, Lt, Ln, ph0, wob))
    occ.append((px_, py_, ray)); chevrons.append(dict(x=round(float(px_)), y=round(float(py_)), amplitude_px=round(A, 1), longueur_onde_px=round(lam), pente=round(4 * A / lam, 2), long_px=round(2 * Lt), epais_px=round(2 * Ln), esp_min_px=round(esp_min, 1)))
FR = PRM.get('froissements', {})
for _ in range(FR.get('nombre', 0)):
    pos = tirer_dans_massif(iT, FR.get('essais', 6), 60.0)                  # froissements : partout dans la coupe, là où il y a un peu de place
    if pos is None: continue
    up, pp = pos; e_loc = e_min_ici(up, pp); px_, py_, et, en = repere(up, pp)
    st, D = float(rng.uniform(*FR['largeur_px'])), float(rng.uniform(*FR['profondeur_px']))
    A, kap, esp_min, essais = pli_sur(st, D, float(rng.uniform(*FR['hauteur'])), float(rng.uniform(*FR['renversement'])), e_loc)
    if A <= 0: continue
    _, _, sm, Lq = pli_tirage(np.zeros(1), np.zeros(1), st, D, A, kap)
    appliquer_local(px_, py_, et, en, 4.5 * sm, Lq, lambda t_, n_: pli_tirage(t_, n_, st, D, A, kap)[:2])
    occ.append((px_, py_, 60.0)); froiss.append(dict(x=round(float(px_)), y=round(float(py_)), largeur_px=round(st), profondeur_px=round(D), hauteur_retenue=round(A, 2), renversement_retenu=round(kap, 2), esp_min_px=round(esp_min, 1)))
# v6 · TOURBILLONS (paquets emmêlés de 0814 : feuilles tordues en S autour d'un point, défauts ±1/2 du réel) : rotation d'angle α(ρ) = α0·exp(−ρ²/2s²)
# autour d'un centre. Chaque cercle tourne sur lui-même : difféomorphisme qui conserve l'aire (jacobien = 1 exactement), inverse explicite (rotation de −α(ρ)).
# Garde-fou : espacement minimal ≥ 2 px (le cisaillement ρ·α'(ρ) rapproche les feuilles), α0 réduit sinon ; placé au plus lâche de 6 positions.
def tourb_tirage(t, n, a0, s_):
    a = a0 * np.exp(-(t * t + n * n) / (2 * s_ * s_)); c_, s2 = np.cos(a), np.sin(a); return c_ * t + s2 * n, -s2 * t + c_ * n
def tourb_sur(a0, s_, e_loc):
    for essai in range(12):
        tt, nn = np.meshgrid(np.arange(-4 * s_, 4 * s_, 0.5), np.arange(-4 * s_, 4 * s_, 0.5)); _, nb = tourb_tirage(tt, nn, a0, s_)
        gy_, gx_ = np.gradient(nb, 0.5); esp_min = e_loc / np.hypot(gx_, gy_).max()
        if esp_min >= EMIN: return a0, esp_min, essai
        a0 *= 0.85
    return 0.0, esp_min, essai
tourb = []; TB = PRM.get('tourbillons', {})
for _ in range(TB.get('nombre', 0)):
    s_ = float(rng.uniform(*TB['rayon_px']))
    pos = tirer_dans_massif(iT, TB.get('essais', 6), 3 * s_)
    if pos is None: continue
    up, pp = pos; e_loc = e_min_ici(up, pp); px_, py_, et, en = repere(up, pp); a0 = float(rng.uniform(*TB['angle_rad'])) * (1 if rng.random() < 0.5 else -1)
    a0, esp_min, ess = tourb_sur(a0, s_, e_loc)
    if abs(a0) < 0.3: continue
    appliquer_local(px_, py_, et, en, 4 * s_, 4 * s_, lambda t_, n_: tourb_tirage(t_, n_, a0, s_))
    occ.append((px_, py_, 3 * s_)); tourb.append(dict(x=round(float(px_)), y=round(float(py_)), rayon_px=round(s_, 1), angle_deg=round(float(np.degrees(a0))), esp_min_px=round(esp_min, 1)))
# v7.3 · CAILLOUX (24/11, relevé sur 8 vraies coupes, etude_degats/ : 18–78 grains clairs / cm², diamètre médian 5 px ; ≥ 15 px : 0–1,8 / cm², jusqu'à 52 px ; un gros caillou
# ÉCARTE les feuilles, qui le contournent en lentille). Ici les cailloux moyens et gros : un trou elliptique (allongé le long des feuilles) s'ouvre, la matière est repoussée
# autour (rayon q -> sqrt(q² − a²), raccord à l'identité entre 3a et 6a : aucune feuille coupée ni croisée). Le trou est une lacune ; le caillou lui-même est peint après
# coup (il n'est PAS dans la vérité : les modèles doivent apprendre à l'ignorer). Les grains fins (< ~8 px) ne déplacent rien : peints seulement.
CX = PRM.get('cailloux', {}); cailloux = []
for _ in range(CX.get('nombre', 0)):
    a_ = float(min(np.exp(rng.normal(np.log(CX.get('rayon_med_px', 8.0)), CX.get('rayon_sigma', 0.5))), CX.get('rayon_max_px', 26.0))); ex_ = float(rng.uniform(1.0, 2.2))
    pos = tirer_dans_massif(iT, 3, 4 * a_ * ex_)
    if pos is None: continue
    up, pp = pos; px_, py_, et, en = repere(up, pp)
    def tir_c(t_, n_, a_=a_, ex_=ex_):
        q = np.sqrt((t_ / ex_) ** 2 + n_ ** 2); qs = np.sqrt(np.maximum(q * q - a_ * a_, 0)); w_ = np.clip((6 * a_ - q) / (3 * a_), 0, 1); f_ = np.where(q > 1e-6, (q - (q - qs) * w_) / np.maximum(q, 1e-6), 0); return t_ * f_, n_ * f_
    appliquer_local(px_, py_, et, en, 6.2 * a_ * ex_, 6.2 * a_, tir_c)
    cyb, cxb = int(py_ + H / 2), int(px_ + W / 2); rb2 = int(a_ * ex_ + 4); by0, by1, bx0, bx1 = max(0, cyb - rb2), min(H, cyb + rb2), max(0, cxb - rb2), min(W, cxb + rb2)
    trou = np.hypot(X[by0:by1, bx0:bx1] - px_, Y[by0:by1, bx0:bx1] - py_) < 0.3
    occ.append((px_, py_, 4 * a_ * ex_)); cailloux.append(dict(x=round(float(px_ + W / 2), 1), y=round(float(py_ + H / 2), 1), rayon_px=round(a_, 1), allongement=round(ex_, 2), angle_deg=round(float(np.degrees(np.arctan2(et[1], et[0]))), 1), _boite=(by0, by1, bx0, bx1), _trou=trou))
T['chevrons_froissements'] = time.time() - t0
# 5c. fissures : ouverture tangentielle (coin) + décalage radial ; lacune = papyrus absent
lacune = np.zeros((H, W), bool); fis = []
for c_ in cailloux:                                                        # trous des cailloux = lacunes (pas de papyrus)
    by0, by1, bx0, bx1 = c_.pop('_boite'); lacune[by0:by1, bx0:bx1] |= c_.pop('_trou')
if PRM['fissures']: rr_, pp_ = np.hypot(X, Y), np.arctan2(Y, X)
def fissure_locale_gpu(R_, P_, Fq, wig, ouv, dec):
    """mêmes formules que fissure_locale, sur la carte graphique (float32)"""
    pc = float(np.radians(Fq['angle_deg'])); rin, rout = Fq['r_rel'][0] * rmax, Fq['r_rel'][1] * rmax
    R = torch.from_numpy(np.ascontiguousarray(R_, np.float32)).cuda(); P = torch.from_numpy(np.ascontiguousarray(P_, np.float32)).cuda(); wg = torch.from_numpy(np.asarray(wig, np.float32)).cuda()
    q = (R / (rmax * 1.5) * 4095).clamp(0, 4095); i0 = q.floor().long().clamp(max=4094); fr = q - i0; wr = wg[i0] * (1 - fr) + wg[i0 + 1] * fr
    dp = torch.remainder(P - pc + np.pi, 2 * np.pi) - np.pi; s_ = R * torch.sin(dp) - wr; front = torch.cos(P - pc) > 0
    lin = ((R - rin) / max(1, rout - rin)).clamp(0, 1); rp = max(30.0, 0.3 * (rout - rin)); c_ = ((R - rin) / rp).clamp(0, 1); env = (c_ ** 2 * (3 - 2 * c_)) * front
    a_ = (ouv * (lin * (2 - lin) if Fq['depuis_bord'] else torch.sin(np.pi * lin) ** 2) * env).clamp(min=1e-3)
    loc = torch.exp(-(s_ / (5 * a_ + 20)) ** 2); sb = s_ - 0.85 * a_ * torch.tanh(s_ / a_) * loc
    Rn = R - dec * 0.5 * torch.tanh(s_ / (a_ + 3)) * loc * (a_ / max(ouv, 1e-6)); Pn = P - (s_ - sb) / R.clamp(min=1); lac = (s_.abs() < 0.8 * a_) & (a_ > 1.2)
    return Rn.cpu().numpy(), Pn.cpu().numpy(), lac.cpu().numpy()
def fissure_locale(R_, P_, Fq, wig, ouv, dec):
    if GPU and R_.size > 20000: return fissure_locale_gpu(R_, P_, Fq, wig, ouv, dec)
    pc = np.radians(Fq['angle_deg']); rin, rout = Fq['r_rel'][0] * rmax, Fq['r_rel'][1] * rmax
    wr = np.interp(R_, np.linspace(0, rmax * 1.5, 4096), wig)
    s_ = R_ * np.sin(np.angle(np.exp(1j * (P_ - pc)))) - wr; front = np.cos(P_ - pc) > 0
    lin = np.clip((R_ - rin) / max(1, rout - rin), 0, 1); rp = max(30.0, 0.3 * (rout - rin)); env = (np.clip((R_ - rin) / rp, 0, 1) ** 2 * (3 - 2 * np.clip((R_ - rin) / rp, 0, 1))) * front
    a_ = np.maximum(ouv * (lin * (2 - lin) if Fq['depuis_bord'] else np.sin(np.pi * lin) ** 2) * env, 1e-3)
    loc = np.exp(-(s_ / (5 * a_ + 20)) ** 2); sb = s_ - 0.85 * a_ * np.tanh(s_ / a_) * loc
    return R_ - dec * 0.5 * np.tanh(s_ / (a_ + 3)) * loc * (a_ / max(ouv, 1e-6)), P_ - (s_ - sb) / np.maximum(R_, 1), (np.abs(s_) < 0.8 * a_) & (a_ > 1.2)
for Fq in PRM['fissures']:
    pc = np.radians(Fq['angle_deg']); rin, rout = Fq['r_rel'][0] * rmax, Fq['r_rel'][1] * rmax
    if not Fq['depuis_bord']: rout = max(rout, rin + 6 * Fq['ouverture_px']); Fq = dict(Fq, r_rel=[Fq['r_rel'][0], rout / rmax])   # fissure interne assez longue pour s'ouvrir sans pli
    wig = Fq['sinuosite_px'] * ndi.gaussian_filter1d(rng.normal(0, 1, 4096), 60)
    # calcul limité à la zone de la fissure (ailleurs le déplacement est nul à 1e-6 près)
    zone = (rr_ > rin - 2) & (rr_ < rout + 40) & (np.abs(rr_ * np.sin(np.angle(np.exp(1j * (pp_ - pc))))) < 8 * Fq['ouverture_px'] + 140) & (np.cos(pp_ - pc) > 0)
    if not zone.any(): continue
    ys_, xs_ = np.nonzero(zone); y0_, y1_, x0_, x1_ = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    R_, P_ = rr_[y0_:y1_, x0_:x1_], pp_[y0_:y1_, x0_:x1_]; ouv, dec = Fq['ouverture_px'], Fq['decalage_px']
    for essai in range(8):                                                     # garde-fou : jacobien local > 0, sinon on réduit décalage puis ouverture
        Rn, Pn, lac = fissure_locale(R_, P_, Fq, wig, ouv, dec)
        Xn, Yn = Rn * np.cos(Pn), Rn * np.sin(Pn); Jl = np.gradient(Xn, axis=1) * np.gradient(Yn, axis=0) - np.gradient(Xn, axis=0) * np.gradient(Yn, axis=1)
        if Jl[1:-1, 1:-1].min() > 0.02: break
        dec *= 0.6; ouv *= 0.85 if essai >= 2 else 1.0
    else:
        continue
    rr_[y0_:y1_, x0_:x1_], pp_[y0_:y1_, x0_:x1_] = Rn, Pn; lacune[y0_:y1_, x0_:x1_] |= lac
    fis.append(dict(Fq, ouverture_retenue=round(ouv, 1), decalage_retenu=round(dec, 1)))
if PRM['fissures']: X, Y = rr_ * np.cos(pp_), rr_ * np.sin(pp_)
T['fissures'] = time.time() - t0
# v7 · PÉTRISSAGE (24/11, retour de Vincent : « tes coupes ne ressemblent pas trop à de vrais rouleaux » — anneaux concentriques, alors que 0306B / 0814
# ont des paquets entiers qui partent en travers et se replient). Dernière déformation physique : on déplace toute la coupe comme une pâte le long d'un
# champ de vitesse lisse SANS divergence (v = rotationnel d'une fonction de courant ψ : l'aire est conservée, rien ne se déchire ni ne se recolle), intégré
# en petits pas (flot = difféomorphisme) : aucune feuille ne se croise, le numéro de tour reste exact. Composition : X(x) = X_v6(K(x)), K = flot arrière.
def champ_lisse(sig):
    """champ aléatoire lisse SANS trace de grille : champ() agrandit une grille grossière (pas sig/6) par interpolation cubique -> dérivées cassées tous les sig/6 px,
    que l'intégration du flot transforme en dents de scie régulières (vu sur les essais 7b–7e). Ici grille au pas sig/24 puis flou gaussien à pleine résolution."""
    f = max(1, int(sig // 24)); g = rng.normal(0, 1, (H // f + 3, W // f + 3)).astype(np.float32); g = ndi.gaussian_filter(g, sig / f)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC); g = cv2.GaussianBlur(g, (0, 0), max(1.0, 2.0 * f)); return g / (g.std() + 1e-6)
PT = PRM.get('petrissage', {})
if PT.get('force', 0) > 0:
    Rr = rmax; psi = np.zeros((H, W), np.float32)
    for srel, w_ in PT.get('echelles', [[0.35, 1.0], [0.16, 0.6], [0.07, 0.3]]):
        psi += w_ * srel * Rr * champ_lisse(srel * Rr)
    rr0 = np.hypot(xx, yy); t_ = np.clip((H / 2 - 4 - rr0) / max(1.0, 0.5 * (H / 2 - 4 - 1.04 * Rr / np.sqrt(k))), 0, 1); psi *= t_ * t_ * (3 - 2 * t_)   # vitesse nulle au bord du canevas
    # v7.1 : le contour d'un vrai rouleau reste plutôt rond même quand l'intérieur est très replié -> fonction de courant atténuée vers le bord
    # (facteur 1 au cœur -> 'bord' au rayon du dernier tour ; reste une fonction de courant : toujours sans divergence)
    if PT.get('bord', 1.0) < 1.0:
        s_ = np.clip((rr0 / Rr - PT.get('bord_debut', 0.45)) / (1.0 - PT.get('bord_debut', 0.45)), 0, 1); psi *= 1 - (1 - PT['bord']) * s_ * s_ * (3 - 2 * s_); del s_
    vy_, vx_ = np.gradient(psi); vx, vy = vy_, -vx_; del psi, vy_, vx_, rr0, t_
    nrm = float(np.sqrt((vx ** 2 + vy ** 2)[np.hypot(xx, yy) < 0.8 * Rr].mean())) + 1e-9; Dp = PT['force'] * Rr; vx *= Dp / nrm; vy *= Dp / nrm
    NP_ = int(np.ceil(float(np.hypot(vx, vy).max()) / PT.get('pas_px', 3.0)))
    if PT.get('ordre', 'base') == 'base':                                # pétrissage PHYSIQUEMENT EN PREMIER (sur la spirale propre) : ondulations, plis et fissures restent nets par-dessus
        # 24/11 (rapide) : le flot Φ est lisse (échelles ≥ 0,2 R) -> on l'intègre sur une grille 4× plus grossière (16× moins de calcul), puis X' = X + D(X) en un seul rééchantillonnage
        fq = int(PT.get('grille', 4)); hq, wq = H // fq + 1, W // fq + 1
        vxq = cv2.resize(vx, (wq, hq), interpolation=cv2.INTER_AREA); vyq = cv2.resize(vy, (wq, hq), interpolation=cv2.INTER_AREA)
        yq, xq = np.mgrid[0:hq, 0:wq].astype(np.float32); PX, PY = xq * fq, yq * fq                      # positions en px pleine résolution (coin du canevas = 0)
        for _ in range(NP_):
            mxq, myq = PX / fq, PY / fq
            PX, PY = PX + cv2.remap(vxq, mxq, myq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / NP_, PY + cv2.remap(vyq, mxq, myq, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / NP_
        DXq, DYq = PX - xq * fq, PY - yq * fq; del PX, PY, vxq, vyq
        mx_, my_ = ((X + W / 2) / fq).astype(np.float32), ((Y + H / 2) / fq).astype(np.float32)
        X, Y = X + cv2.remap(DXq, mx_, my_, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0), Y + cv2.remap(DYq, mx_, my_, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        del mx_, my_, DXq, DYq
        vx = None
    KX, KY = xx.copy(), yy.copy()
    if vx is not None:
      for _ in range(NP_):
          mx_, my_ = KX + W / 2, KY + H / 2
          KX, KY = KX + cv2.remap(vx, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / NP_, KY + cv2.remap(vy, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / NP_
      del vx, vy; mx_, my_ = KX + W / 2, KY + H / 2; del KX, KY
      X = cv2.remap(X, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e6); Y = cv2.remap(Y, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e6)
      lacune = cv2.remap(lacune.astype(np.uint8), mx_, my_, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
      Wf = cv2.remap(Wf, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0); del mx_, my_
    PT = dict(PT, pas=NP_, deplacement_px=round(Dp, 1))
T['petrissage'] = time.time() - t0
# v7.4 · PLIS FRANCS MULTIPLES (24/11, test à l'aveugle n° 3 : « les vrais ont des formes très complexes et les tiennes sont plutôt régulières »). Vu sur 0306B / 0139 :
# plusieurs lignes de pliure qui ne passent pas par le centre ; un pan entier de feuilles y change brutalement de direction. Ici, pour chaque pli : une droite (centre c,
# direction t, normale n) ; les points à distance d < D de la droite glissent LE LONG de la droite de u = k·(D − |d|)·fenêtre(s) : cisaillement en « toit », plié sur la
# droite (les feuilles qui la traversent font un angle net), continu partout, jacobien 1 + k·(D − |d|)·fenêtre'(s) > 0 (fenêtre longue devant D). Rien n'est coupé ni croisé.
PF = PRM.get('plis_francs', []); plis_fr = []
for q_ in PF:
    cx_, cy_ = q_['centre_rel'][0] * R, q_['centre_rel'][1] * R; a_ = np.radians(q_['angle_deg']); tx_, ty_ = np.cos(a_), np.sin(a_); D_ = q_['demi_largeur_rel'] * R; L_ = q_['longueur_rel'] * R; k_ = q_['pente']
    k_ = float(np.sign(k_) * min(abs(k_), 0.45 * L_ / max(D_, 1e-6)))                                     # garde-fou : k·D·max|fenêtre'| ≈ k·D/L < 0,5
    s_ = (X - cx_) * tx_ + (Y - cy_) * ty_; d_ = -(X - cx_) * ty_ + (Y - cy_) * tx_; u_ = (k_ * np.maximum(D_ - np.abs(d_), 0) * np.exp(-(s_ / L_) ** 2)).astype(np.float32)
    X = (X + u_ * tx_).astype(np.float32); Y = (Y + u_ * ty_).astype(np.float32); plis_fr.append(dict(q_, pente_retenue=round(k_, 2))); del s_, d_, u_
T['plis_francs'] = time.time() - t0
# v7.3 · APLATISSEMENT « EN AMANDE » + FISSURE AXIALE (24/11, étude des vraies coupes demandée par Vincent : « la fissure a lieu en général dans une épingle, dans un angle »).
# Vu sur 0211 z8390 : cœur en amande à deux pointes ; de chaque pointe part la LIGNE DES ÉPINGLES (toutes les feuilles y font un V serré) et la longue déchirure suit
# cette ligne, avec des coins d'air aux sommets et des feuilles décalées d'une lèvre à l'autre. Ici : chaque tour = deux arcs de cercle (centres à ∓h de l'axe) qui se
# rejoignent en pointe sur l'axe -> carte continue, pliée sur l'axe (les feuilles ne se croisent pas, le numéro de tour reste exact). La fissure s'ouvre le long de l'axe :
# largeur variable, fermée par endroits, décalage des feuilles proportionnel à l'ouverture ; la partie ouverte est une lacune (pas de papyrus).
AM = PRM.get('amande', {})
if AM.get('h_rel', 0) > 0:
    aa_ = np.radians(AM.get('angle_deg', 0.0)); Xa = (np.cos(aa_) * X + np.sin(aa_) * Y).astype(np.float32); Ya = (-np.sin(aa_) * X + np.cos(aa_) * Y).astype(np.float32)
    ha = np.float32(AM['h_rel'] * R); sg_ = np.where(Ya >= 0, 1.0, -1.0).astype(np.float32); FI = AM.get('fissure', {}); info_am = dict(AM)
    if FI.get('largeur_px', 0) > 0:
        xm_ = float(np.sqrt(R * R + 2 * R * ha)) * 1.3; n1 = 4096; x1 = np.linspace(-xm_, xm_, n1); pas1 = x1[1] - x1[0]
        z1 = ndi.gaussian_filter1d(rng.normal(0, 1, n1), FI.get('corr_px', 220) / pas1); z1 /= z1.std() + 1e-9
        cote = np.where(x1 < 0, FI.get('gauche', 1.0), FI.get('droite', 1.0)); tap = np.clip((np.abs(x1) - FI.get('debut_rel', 0.06) * xm_) / (0.05 * xm_), 0, 1)
        w1 = FI['largeur_px'] * np.clip(z1 + FI.get('biais', 0.2), 0, None) * cote * tap                     # largeur de la fissure le long de l'axe (0 = fermée)
        if FI.get('coin_px', 0) > 0:                                        # v7.3b : grande déchirure qui s'ouvre EN COIN jusqu'au bord, d'un seul côté (vu sur 0306B z12432, 0211)
            xe_ = float(np.sqrt(R * R + 2 * R * ha)); sgn_ = FI.get('coin_cote', 1.0); u_ = np.clip((sgn_ * x1 - FI.get('coin_debut_rel', 0.45) * xe_) / ((1 - FI.get('coin_debut_rel', 0.45)) * xe_), 0, 1.3)
            w1 = w1 + FI['coin_px'] * u_ ** 1.6 * (1 + 0.25 * z1)
        wA = np.interp(Xa, x1, w1).astype(np.float32); aY = np.abs(Ya); lacune |= (aY < wA / 2) & (wA > 1.0)
        dec_ = FI.get('decalage_px', 0.0) * np.clip(wA / max(FI['largeur_px'], 1e-6), 0, 1.5) * (wA > 1.0); Ya = sg_ * np.maximum(aY - wA / 2, 0); Xa = Xa + sg_ * 0.5 * dec_
        info_am['fissure_ouverte_pct'] = round(float((w1 > 1.0).mean()) * 100, 1); info_am['fissure_largeur_max_px'] = round(float(w1.max()), 1); del wA, aY, dec_
    aY = np.abs(Ya); rho_ = np.sqrt(Xa * Xa + (aY + ha) ** 2) - ha; be_ = np.arctan2(aY + ha, Xa); b0_ = np.arctan2(ha, np.sqrt(np.maximum((rho_ + ha) ** 2 - ha * ha, 1e-6)))
    ph_ = (be_ - b0_) / np.maximum(np.pi - 2 * b0_, 1e-6) * np.pi; Xa, Ya = rho_ * np.cos(ph_), sg_ * rho_ * np.sin(ph_); del aY, rho_, be_, b0_, ph_, sg_
    X = (np.cos(aa_) * Xa - np.sin(aa_) * Ya).astype(np.float32); Y = (np.sin(aa_) * Xa + np.cos(aa_) * Ya).astype(np.float32); del Xa, Ya
    AM = info_am
T['amande'] = time.time() - t0
# 5d. ovale léger (inverse de l'étirement) puis coordonnées polaires de base
xr, yr = np.cos(al) * X + np.sin(al) * Y, -np.sin(al) * X + np.cos(al) * Y; X, Y = xr * np.sqrt(k), yr / np.sqrt(k)
rb, pb_ = np.hypot(X, Y), np.arctan2(Y, X); del xr, yr
mx = (rb / dr).astype(np.float32); my = ((pb_ + np.pi) / (2 * np.pi) * NPH).astype(np.float32)
U = cv2.remap(tab, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(uu[-1, 0]))
Fz = U - pb_ / (2 * np.pi)
T['champ'] = time.time() - t0
# ---------- 6. contrôles : jacobien de la carte final -> base, gradient, ordre ----------
J = np.gradient(X, axis=1) * np.gradient(Y, axis=0) - np.gradient(X, axis=0) * np.gradient(Y, axis=1)
def grad_enroule(Fz):
    w = lambda d: d - np.round(d)
    gx_ = np.zeros_like(Fz); gy_ = np.zeros_like(Fz); gx_[:, 1:-1] = w(Fz[:, 2:] - Fz[:, :-2]) / 2; gy_[1:-1, :] = w(Fz[2:, :] - Fz[:-2, :]) / 2
    return gy_, gx_
gy, gx = grad_enroule(Fz); gn = np.hypot(gx, gy)
dedans = (U > 0) & (U < N + pb_ / (2 * np.pi) + 0.02)              # entre le cœur et le dernier tour (le bord extérieur EST la dernière feuille)
dedans = ndi.binary_erosion(dedans, iterations=1)
# ---------- 7. raster ----------
# v6 : TOUTES les lignes sont tirées de W par PASSAGES À ZÉRO (plus de squelette d'une bande seuillée) : entre deux pixels voisins (4-voisinage)
# du MÊME tour (même arrondi de F), si G = (F − n)/|∇F| ∓ δ change de signe, on marque le pixel le plus proche du niveau. Deux niveaux distincts
# donnent deux courbes distinctes ; deux niveaux à < 1 px tombent sur le même pixel (une ligne, jamais de pointillés).
dist = np.abs(Fz - np.round(Fz)) / np.maximum(gn, 1e-4)
nF = np.round(Fz); sd = (Fz - nF) / np.maximum(gn, 1e-4); el = 1 / np.maximum(gn, 1e-4)
def passages(G, ok):
    if GPU:
        Gt = torch.from_numpy(np.ascontiguousarray(G, np.float32)).cuda(); nt = torch.from_numpy(np.ascontiguousarray(nF, np.float32)).cuda(); ot = torch.from_numpy(np.ascontiguousarray(ok)).cuda(); mt = torch.zeros(G.shape, dtype=torch.bool, device='cuda')
        for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))), ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
            ga, gb = Gt[sl_a], Gt[sl_b]; c = (torch.sign(ga) != torch.sign(gb)) & (nt[sl_a] == nt[sl_b]) & ot[sl_a] & ot[sl_b]
            pa = c & (ga.abs() <= gb.abs()); mt[sl_a] |= pa; mt[sl_b] |= c & ~pa
        return mt.cpu().numpy()
    m = np.zeros(G.shape, bool)
    for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))), ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        ga, gb = G[sl_a], G[sl_b]; c = (np.sign(ga) != np.sign(gb)) & (nF[sl_a] == nF[sl_b]) & ok[sl_a] & ok[sl_b]
        pa = c & (np.abs(ga) <= np.abs(gb)); m[sl_a] |= pa; m[sl_b] |= c & ~pa
    return m
lignes = passages(sd, dedans)
tour = np.where(lignes, nF.astype(np.int32), 0)                       # sur la feuille F = n exactement : n = numéro de tour
visibles = lignes & ~lacune
# SOUS-FEUILLES, ÉPAISSISSEMENTS et FRANGES (retour de Vincent, 23/11 soir) : la feuille se dédouble par endroits de ±δ (δ < 30 % de l'écart au tour voisin :
# jamais de contact entre tours), δ = max(dédoublement, épaississement) ; la VÉRITÉ (lignes, tour) reste la ligne centrale, « papier » = ce qu'on donne au peintre.
SF = PRM.get('sous_feuilles', {})
if SF:
    dlt = np.clip(champ(SF.get('arc_px', 60)) * SF.get('ecart_px', 2.5) - SF.get('seuil', 0.6) * SF.get('ecart_px', 2.5), 0, None)
    dlt = np.maximum(dlt, np.clip(champ(40) * SF.get('epaisseur_px', 0.6), 0, None))
    dlt = np.minimum(dlt, 0.3 * el)
    dlt = np.where(dlt >= SF.get('dlt_min', 1.0), dlt, 0.0)                                  # dédoublement dessiné seulement si les deux couches sont à ≥ 2 px (sinon une seule ligne : pas de bande blanche dans les massifs)
    papier = passages(sd - dlt, dedans) | passages(sd + dlt, dedans)
    frange = (champ(12) > SF.get('franges_seuil', 2.2)) & (np.abs(sd) < 4)   # petites interruptions de la feuille
    papier &= ~frange & ~lacune
else:
    papier = visibles
T['raster'] = time.time() - t0
esp = (1 / np.maximum(gn, 1e-4))[dedans & ~lacune]
_okJ = dedans & ~ndi.binary_dilation(lacune, iterations=3)                 # v7.3 : la fissure ouverte (lacune) et ses lèvres ne comptent pas (carte volontairement discontinue)
info = dict(nom=NOM, version='v7', petrissage=PT, amande=AM, cailloux=cailloux, plis_francs=plis_fr, valide=bool((J[_okJ] <= 0).sum() == 0),   # valide = aucun pixel à jacobien ≤ 0 : à exiger pour les données d'entraînement
            diametre_mm=DMM, graine=SEED, taille_px=[H, W], tours=N,
            ecart_tours_px=dict(p5=round(float(np.percentile(esp, 5)), 1), p25=round(float(np.percentile(esp, 25)), 1), mediane=round(float(np.median(esp)), 1),
                                p75=round(float(np.percentile(esp, 75)), 1), p95=round(float(np.percentile(esp, 95)), 1)),
            controle=dict(jacobien_min=round(float(J[_okJ].min()), 4), part_jacobien_negatif=round(float((J[_okJ] <= 0).mean()) * 100, 5),
                          min_gradient=round(float(gn[dedans].min()), 4), lacune_pct=round(float(lacune[dedans].mean()) * 100, 2)),
            nb_ecrasements=len(zones), nb_plis=len(plis), nb_fissures=len(fis), nb_chevrons=len(chevrons), nb_froissements=len(froiss), nb_tourbillons=len(tourb),
            massifs_pct=round(float(Wf[dedans].mean()) * 100, 1), plis=plis, chevrons=chevrons, froissements=froiss, tourbillons=tourb, parametres=PRM, temps_s={k_: round(v, 1) for k_, v in T.items()})
np.savez_compressed(f'{O}/{NOM}.npz', lignes=(lignes * 255).astype(np.uint8), visibles=(visibles * 255).astype(np.uint8), papier=(papier * 255).astype(np.uint8), tour=np.clip(tour, 0, 65535).astype(np.uint16),
                    dist=np.clip(np.where(dedans, dist, 255), 0, 255).astype(np.uint8), dist10=np.clip(np.where(dedans, dist * 10, 255), 0, 255).astype(np.uint8),
                    dedans=dedans, lacune=lacune & dedans, jneg=(J <= 0) & dedans)
json.dump(info, open(f'{O}/{NOM}.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k_: v for k_, v in info.items() if k_ not in ('parametres', 'plis')}, ensure_ascii=False, default=float))
