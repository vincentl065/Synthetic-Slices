"""24/11 · CASSURES SUR TOUT LE ROULEAU, DANS LE TRACÉ (Vincent : « il faut vraiment simuler une cassure sur les tracés » ; les feuilles cassées « doivent constituer ses
bords, de manière irrégulière mais cohérente »). Rien n'est posé sur l'image : on modifie le tracé donné au peintre, qui peint lui-même les lèvres déchirées.
  1. chemins de fissure : ceux de la forme (lacune) + de NOUVELLES fissures comme sur 0306B : elles longent les feuilles (frontière entre deux paquets) puis
     sautent en travers, largeur très variable (fermée par endroits, jusqu'à quelques dizaines de px), l'une part du bord ;
  2. autour de ces chemins : chaque feuille recule d'une longueur différente (escalier), des paquets entiers reculent ensemble (langues), pointillés avant la rupture ;
  3. quelques bouts de trait restent dans le vide (bouts de page).
Sortie : même fichier de forme avec vis (tracé du peintre) cassé (vis0 = d'origine) ; lig (VÉRITÉ) inchangée et continue ; « cassure » = zone rompue dans l'image.
    python casser_trace_entier.py <geo.npz> <sortie.npz> [graine=0] [nb_fissures=3] [largeur_max_px=45]"""
import sys, json, numpy as np, cv2
SRC, OUT = sys.argv[1], sys.argv[2]; G = int(sys.argv[3]) if len(sys.argv) > 3 else 0; NF = int(sys.argv[4]) if len(sys.argv) > 4 else 3; WMAX = float(sys.argv[5]) if len(sys.argv) > 5 else 45.0
rng = np.random.default_rng(G); z = dict(np.load(SRC)); meta = json.loads(str(z["meta"])); H, W = meta["taille"]
bits = lambda k: np.unpackbits(z[k], axis=-1)[:, :W] > 0
vis, ded, lig = bits("vis"), bits("ded"), bits("lig"); lac = bits("lac").astype(np.uint8)
U = cv2.resize(z["U2"], (W, H), interpolation=cv2.INTER_LINEAR); cle = np.floor(U - z["ph8"].astype(np.float32) / 256 + 0.5).astype(np.int64)
def lisse(sig):
    v = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (H // 8 + 2, W // 8 + 2)).astype(np.float32), (0, 0), sig / 8), (W, H), interpolation=cv2.INTER_LINEAR); return (v - v.mean()) / (v.std() + 1e-9)
# fissures de la forme (sans les trous compacts des cailloux)
k, lab, st, _ = cv2.connectedComponentsWithStats(lac, 8); garde = np.zeros(k, bool)
for i in range(1, k): garde[i] = st[i][4] > 2500 or max(st[i][2], st[i][3]) > 5 * max(1, min(st[i][2], st[i][3]))
fis = garde[lab].astype(np.uint8); trous = (lac > 0) & (fis == 0)
# 1. nouvelles fissures : marche qui longe les feuilles puis saute en travers
Uq = cv2.GaussianBlur(cv2.resize(z["U2"], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3); gy, gx = np.gradient(Uq); dq = ded[::4, ::4][:Uq.shape[0], :Uq.shape[1]]
bord = dq & ~cv2.erode(dq.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool); yb, xb = np.nonzero(bord); yi, xi = np.nonzero(cv2.erode(dq.astype(np.uint8), np.ones((61, 61), np.uint8)))
def marche(y, x, long, wmax, travers_d_abord):
    pts = []; mode = "travers" if travers_d_abord else "long"; dT = (220, 800) if travers_d_abord else (40, 160); dL = (30, 130) if travers_d_abord else (300, 950); reste = rng.uniform(*dT) if mode == "travers" else rng.uniform(*dL); sl_, st_ = rng.choice([-1, 1]), (-1 if travers_d_abord else rng.choice([-1, 1])); ang = None; s = 0.0
    while s < long:
        qy, qx = int(np.clip(y / 4, 0, Uq.shape[0] - 1)), int(np.clip(x / 4, 0, Uq.shape[1] - 1))
        if not dq[qy, qx] and s > 40: break
        g = np.array([gx[qy, qx], gy[qy, qx]]); g = g / (np.linalg.norm(g) + 1e-9); t = np.array([-g[1], g[0]])
        v = g * st_ if mode == "travers" else t * sl_; cible = np.arctan2(v[1], v[0]) + rng.normal(0, 0.35 if mode == "travers" else 0.10)
        ang = cible if ang is None else ang + 0.22 * np.angle(np.exp(1j * (cible - ang)))
        pas = 10.0; y, x = y + pas * np.sin(ang), x + pas * np.cos(ang); s += pas; reste -= pas; pts.append((x, y, 1.0 if mode == "travers" else 0.4))
        if reste <= 0:
            mode = "long" if mode == "travers" else "travers"; reste = rng.uniform(*dT) if mode == "travers" else rng.uniform(*dL); sl_ = rng.choice([-1, 1]) if travers_d_abord else sl_
    if len(pts) < 8: return
    n = len(pts); prof = np.clip(cv2.GaussianBlur(rng.normal(0, 1, (n, 1)).astype(np.float32), (0, 0), sigmaX=1, sigmaY=9).ravel() * 1.4 + 0.45, 0, 1.6)
    fin = np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(6.0, 0.12 * n); prof = prof * np.clip(fin, 0, 1) if not travers_d_abord else prof * np.clip((n - 1 - np.arange(n)) / max(6.0, 0.12 * n), 0, 1) * np.linspace(1.6, 0.8, n)
    for i in range(n - 1):
        e = int(round(wmax * prof[i] * pts[i][2]))
        if e >= 1: cv2.line(neuves, (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1])), 1, e, cv2.LINE_8)
neuves = np.zeros((H, W), np.uint8)
for j in range(NF):
    du_bord = j == 0 and len(yb) > 0
    if du_bord: q = rng.integers(len(yb)); y0, x0 = yb[q] * 4.0, xb[q] * 4.0
    elif len(yi): q = rng.integers(len(yi)); y0, x0 = yi[q] * 4.0, xi[q] * 4.0
    else: continue
    marche(y0, x0, (rng.uniform(0.55, 0.95) * max(H, W)) if du_bord else rng.uniform(1200, 4200) * min(1.0, max(H, W) / 5000), WMAX * (rng.uniform(1.5, 2.2) if du_bord else rng.uniform(0.5, 1.3)), du_bord)   # la première = grande déchirure qui part du bord et traverse une bonne partie du rouleau (0306B, 0211)
chemin = ((fis > 0) | (neuves > 0)).astype(np.uint8)
# 2. recul feuille par feuille + par paquets, pointillés
d = cv2.distanceTransform(1 - chemin, cv2.DIST_L2, 5); cote = (lisse(250) > 0).astype(np.int64)
u = (((cle * 73856093) ^ (cote * 19349663) ^ (G * 83492791)) & 0xFFFF).astype(np.float32) / 65535.0
r = np.where(u < 0.45, 5 + u * 25, np.where(u < 0.85, 16 + (u - 0.45) * 50, 36 + (u - 0.85) * 800)) * np.clip(1.7 + 1.0 * lisse(180), 0.4, 3.5)
r *= np.clip(0.15 + cv2.GaussianBlur(chemin.astype(np.float32), (0, 0), 25) * 14, 0.15, 1.0)                  # là où la fissure est fermée ou très fine, les feuilles reculent à peine
vide = (d < r) | (chemin > 0)
u2 = (((cle * 2654435761) ^ (cote * 40503)) & 0xFFFF).astype(np.float32) / 65535.0; bande = 20 + 70 * u2
tirets = cv2.resize(rng.random((H // 7 + 2, W // 7 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
coupe = (~vide) & (d < r + bande) & (tirets < 0.55 * (1 - (d - r) / bande)) & (d < 200)
casse = vis & ~vide & ~coupe
# 3. bouts de trait isolés dans le vide
ys, xs = np.nonzero(vide & (chemin == 0) & (d > 6)); nb = 0
if len(ys):
    for _ in range(int(len(ys) / 5000) + 2):
        q = rng.integers(len(ys)); cy, cx = int(ys[q]), int(xs[q]); R = int(rng.uniform(16, 44))
        if not (R <= cy < H - R and R <= cx < W - R): continue
        p = vis[cy - R:cy + R, cx - R:cx + R].astype(np.uint8) * 255; mk = np.zeros((2 * R, 2 * R), np.uint8); cv2.ellipse(mk, (R, R), (R - 1, max(8, int(R * rng.uniform(0.3, 0.7)))), float(rng.uniform(0, 180)), 0, 360, 255, -1)
        M = cv2.getRotationMatrix2D((R, R), float(rng.normal(0, 28)), 1.0); p = cv2.warpAffine(p & mk, M, (2 * R, 2 * R), flags=cv2.INTER_NEAREST) > 0
        casse[cy - R:cy + R, cx - R:cx + R] |= p & vide[cy - R:cy + R, cx - R:cx + R]; nb += 1
lac2 = ((vide & ded) | trous)
# 24/11, Vincent : « les traits sont supposés ne pas être impactés et être continus » -> la VÉRITÉ (lig) reste intacte et continue à travers les cassures ; seul le tracé
# donné au peintre (vis) est cassé. lac0 = lacune de la forme ; « cassure » = zone où l'image est rompue (information, pas un masque à retirer de la vérité).
z["vis0"] = z["vis"]; z["vis"] = np.packbits(casse, axis=-1); z["lac0"] = z["lac"]; z["cassure"] = np.packbits(lac2, axis=-1)
meta["cassures"] = dict(graine=G, fissures_neuves=NF, largeur_max_px=WMAX, bouts_de_trait=nb, lacune_pct=round(float(lac2.mean()) * 100, 2), trace_retire_pct=round(100 * (1 - casse.sum() / max(vis.sum(), 1)), 1)); z["meta"] = json.dumps(meta)
np.savez_compressed(OUT, **z); print(meta["nom"], meta["cassures"])
e = 1400 / max(H, W); v = np.zeros((H, W, 3), np.uint8); v[casse] = 255; v = cv2.resize(v, None, fx=e, fy=e, interpolation=cv2.INTER_AREA); m = cv2.resize(lac2.astype(np.uint8) * 255, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_AREA) > 40
v[m] = (0, 159, 230); cv2.imwrite(OUT[:-4] + "_apercu.jpg", np.clip(v.astype(np.float32) * 2.2, 0, 255).astype(np.uint8))
