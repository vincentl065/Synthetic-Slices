"""25/11 · ton et netteté d'un vrai scan À PARTIR D'UN PROFIL (201 centiles de gris + netteté), sans la vraie coupe : même calcul que ton_reel.py.
Les vraies coupes ne quittent pas le PC de Vincent ; seuls ces 202 nombres par coupe sont envoyés sur une machine louée (profils_ton.json, fait par profil_ton.py).
    python ton_profil.py <nous.png> <profils_ton.json> <nom du profil> <sortie.png>"""
import sys, json, numpy as np, cv2
def dedans(a): return cv2.erode((cv2.blur((a > 0).astype(np.float32), (41, 41)) > 0.99).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
def nettete(a, m): return float(np.abs(cv2.Laplacian(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0), cv2.CV_32F))[m][::7].mean())
a = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE); P = json.load(open(sys.argv[2], encoding="utf-8"))[sys.argv[3]]; ma = dedans(a)
q = np.linspace(0, 100, 201); qa = np.percentile(a[ma][::5], q); qv = np.array(P["centiles"]); cible = P["nettete"]
lut = np.interp(np.arange(256), qa + np.arange(201) * 1e-6, qv).astype(np.float32); b = lut[a]; best = (1e9, 0, b)
for sg in (0, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4):
    c = cv2.GaussianBlur(b, (0, 0), sg) if sg > 0 else b; e = abs(nettete(c, ma) - cible)
    if e < best[0]: best = (e, sg, c)
o = np.where(a > 0, np.clip(np.round(best[2]), 1, 255), 0).astype(np.uint8); cv2.imwrite(sys.argv[4], o)
print(sys.argv[4], "profil", sys.argv[3], "flou", best[1], "nettete", round(nettete(o, ma), 2), "cible", round(cible, 2))
