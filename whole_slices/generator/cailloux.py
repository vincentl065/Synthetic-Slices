"""24/11 · CAILLOUX dans l'image (jamais dans la vérité : Vincent, « un artefact aléatoire que le générateur peut faire apparaître dans des endroits adéquats, mais ignoré
par le tracé »). Proportions tirées du relevé sur 8 vraies coupes (etude_degats/mesurer_cailloux.py) :
  - GROS / MOYENS : ceux de la forme (meta['cailloux'] : centre, rayon, allongement, angle ; les feuilles s'écartent déjà autour) -> polygone anguleux très clair, bord net ;
  - GRAINS FINS : DENS par cm² (réel 18–78), diamètre médian 5 px (2,5–9), clarté ≈ 244, posés au hasard dans le rouleau (surtout dans la matière ; PART_VIDE dans un vide).
À appliquer APRÈS la peinture et AVANT le flou de mouvement (le flou est un défaut du scan : il étale aussi les cailloux).
    python cailloux.py <jeu.npz | geo.npz> <image.png> <sortie.png> [grains_par_cm2=40] [graine=0]"""
import sys, json, numpy as np, cv2
SRC, IMG, OUT = sys.argv[1:4]; DENS = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0; G = int(sys.argv[5]) if len(sys.argv) > 5 else 0; rng = np.random.default_rng(G); UM = 9.4
z = np.load(SRC); m = json.loads(str(z["meta"])); H, W = m["taille"]; ded = np.unpackbits(z["ded"], axis=-1)[:, :W] > 0; img = cv2.imread(IMG, 0).astype(np.float32)
def pierre(cx, cy, ra, rb, ang, clair):
    """polygone anguleux (6–10 sommets) dans une ellipse (ra le long de ang, rb en travers), rendu ×4 puis réduit : bord net d'un pixel, léger relief interne"""
    n = int(rng.integers(6, 11)); th = np.sort(rng.uniform(0, 2 * np.pi, n)); rr = rng.uniform(0.62, 1.0, n); S = 4; R = int(max(ra, rb) + 3); c = R * S
    px = ra * rr * np.cos(th); py = rb * rr * np.sin(th); ca, sa = np.cos(ang), np.sin(ang); pts = np.stack([c + S * (ca * px - sa * py), c + S * (sa * px + ca * py)], 1).astype(np.int32)
    mk = np.zeros((2 * c, 2 * c), np.uint8); cv2.fillPoly(mk, [pts], 255); al = cv2.resize(mk, (2 * R, 2 * R), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    tex = clair + cv2.GaussianBlur(rng.normal(0, 1, (2 * R, 2 * R)).astype(np.float32), (0, 0), 1.2) * 9 - 10 * (rng.random() < 0.3)
    y0, x0 = int(round(cy)) - R, int(round(cx)) - R; ya, yb, xa, xb = max(0, y0), min(H, y0 + 2 * R), max(0, x0), min(W, x0 + 2 * R)
    if yb <= ya or xb <= xa: return
    a_ = al[ya - y0:yb - y0, xa - x0:xb - x0]; img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a_) + np.clip(tex[ya - y0:yb - y0, xa - x0:xb - x0], 0, 255) * a_
gros = m.get("cailloux", [])
for c in gros:
    pierre(c["x"], c["y"], c["rayon_px"] * c["allongement"] * 0.9, c["rayon_px"] * 0.9, np.radians(c["angle_deg"]), rng.uniform(232, 252))
    for _ in range(int(rng.integers(0, 4)) if c["rayon_px"] > 12 else 0):                                   # satellites des gros (vu sur 0306B, 0814)
        d_ = c["rayon_px"] * rng.uniform(1.3, 2.6); t_ = rng.uniform(0, 2 * np.pi); r_ = rng.uniform(1.5, 4); pierre(c["x"] + d_ * np.cos(t_), c["y"] + d_ * np.sin(t_), r_, r_ * rng.uniform(0.6, 1), rng.uniform(0, np.pi), rng.uniform(215, 248))
surf = ded.sum() * (UM * 1e-4) ** 2; N = int(rng.poisson(DENS * surf)); ys, xs = np.nonzero(ded); k = rng.integers(0, len(ys), N); d = np.clip(np.exp(rng.normal(np.log(4.6), 0.33, N)), 2.5, 9.0)
for i in range(N): pierre(xs[k[i]] + rng.uniform(-0.5, 0.5), ys[k[i]] + rng.uniform(-0.5, 0.5), d[i] / 2 * rng.uniform(1.0, 1.5), d[i] / 2 * rng.uniform(0.7, 1.0), rng.uniform(0, np.pi), rng.uniform(212, 250))
cv2.imwrite(OUT, np.where(cv2.imread(IMG, 0) > 0, np.clip(img, 1, 255), 0).astype(np.uint8)); print(OUT, "gros", len(gros), "| grains fins", N, f"({DENS:.0f} / cm2 sur {surf:.1f} cm2)")
