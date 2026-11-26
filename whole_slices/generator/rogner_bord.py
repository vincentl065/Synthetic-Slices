"""24/11 · BORD DU ROULEAU : la peinture laisse une large bande grise autour du rouleau (masque dilaté de 60 px), qui nous trahit au test à l'aveugle ; les vrais ont un
liseré fin. On ne garde qu'un liseré de LARG px (± bruit lent, borné pour ne jamais mordre dans le rouleau), fondu sur 3 px.
    python rogner_bord.py <geo_ou_jeu.npz> <image.png> <sortie.png> [largeur_px=14] [graine=0]"""
import sys, numpy as np, cv2
from scipy import ndimage as ndi
SRC, IMG, OUT = sys.argv[1:4]; LARG = float(sys.argv[4]) if len(sys.argv) > 4 else 14.0; G = int(sys.argv[5]) if len(sys.argv) > 5 else 0
a = cv2.imread(IMG, 0); h, w = a.shape; z = np.load(SRC); ded = ndi.binary_fill_holes((np.unpackbits(z["ded"], axis=-1)[:, :w] > 0) & ~ndi.binary_erosion(np.unpackbits(z["lac"], axis=-1)[:, :w] > 0, iterations=3)).astype(np.uint8)
rng = np.random.default_rng(G); b = np.clip(cv2.resize(rng.normal(0, 1, (h // 160 + 2, w // 160 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), -1.5, 2.0)
d = cv2.distanceTransform(1 - ded, cv2.DIST_L2, 5); m = np.clip((LARG + 0.35 * LARG * b - d) / 3.0 + 0.5, 0, 1)
cv2.imwrite(OUT, (a * m).astype(np.uint8)); print(OUT, "bord rogné à", LARG, "px")
