"""24/11 · FLOU DE MOUVEMENT PAR PLAGE (idée et réglage de Vincent : « il n'y a pas une seule bonne réponse, il peut y avoir de tout, mais le plus courant serait de l'ordre
du 9 px fort »). Chaque plage écrasée tire SES réglages : longueur (5–14 px, mode 9), force (0,5–1, mode 0,85), une direction fixe ou deux directions croisées (60 %),
angle au hasard (défaut du scan : indépendant des feuilles). Longueur plafonnée à 12 px le 24/11 midi (à 12–14 px fort : flou visible). Ne déplace aucune feuille ; à appliquer dans les plages déjà marquées illisibles.
    python flou_mouvement2.py <image.png> <plages.png> <sortie.png> [graine=0]"""
import sys, numpy as np, cv2
from scipy import ndimage as ndi
IMG, PLG, OUT = sys.argv[1:4]; G = int(sys.argv[4]) if len(sys.argv) > 4 else 0; rng = np.random.default_rng(G)
img = cv2.imread(IMG, 0).astype(np.float32); H, W = img.shape; pl = cv2.imread(PLG, 0).astype(np.float32) / 255
lab, n = ndi.label(pl > 0.3); lab = ndi.grey_dilation(lab, size=(41, 41))                                 # une étiquette par plage, un peu élargie pour couvrir le bord fondu
lg = np.zeros(n + 1, np.float32); fo = np.zeros(n + 1, np.float32); an = rng.uniform(0, np.pi, n + 1).astype(np.float32); cr = np.zeros(n + 1, np.float32); ec = rng.uniform(1.0, 1.45, n + 1).astype(np.float32)
lg[1:] = rng.triangular(5, 9, 12, n); fo[1:] = rng.triangular(0.5, 0.85, 1.0, n); cr[1:] = rng.random(n) < 0.6
LG, FO, CR = lg[lab], fo[lab], cr[lab]; c1, s1 = np.cos(an[lab]), np.sin(an[lab]); c2, s2 = np.cos(an[lab] + ec[lab]), np.sin(an[lab] + ec[lab])
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); a1 = np.zeros((H, W), np.float32); a2 = np.zeros((H, W), np.float32); N = 11
for k in np.linspace(-0.5, 0.5, N):
    k = np.float32(k); a1 += cv2.remap(img, xx + k * LG * c1, yy + k * LG * s1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT); a2 += cv2.remap(img, xx + k * LG * c2, yy + k * LG * s2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
a1 /= N; a2 /= N; fl = np.where(CR > 0, np.maximum(a1, a2) * 0.55 + (a1 + a2) * 0.225, a1)
f_ = 16; g_ = cv2.GaussianBlur(rng.normal(0, 1, (H // f_ + 2, W // f_ + 2)).astype(np.float32), (0, 0), 60 / f_); g_ = cv2.GaussianBlur(cv2.resize(g_, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 2 * f_); g_ /= g_.std() + 1e-6
plaques = cv2.GaussianBlur((0.55 + 0.45 * (g_ > -0.3)).astype(np.float32), (0, 0), 5)                    # comme sur la planche validée par Vincent : pleine force par plaques seulement
al = cv2.GaussianBlur(pl, (0, 0), 6) * FO * plaques; out = img * (1 - al) + fl * al; cv2.imwrite(OUT, np.where(img > 0, np.clip(out, 1, 255), 0).astype(np.uint8))
print(OUT, n, "plages | longueurs", np.round(np.sort(lg[1:]), 1).tolist(), "| forces", np.round(np.sort(fo[1:]), 2).tolist(), "| croisees", int(cr.sum()))
