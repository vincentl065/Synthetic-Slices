"""26/11 · Sort la réponse brute du modèle sur des coupes entières (réduite d'un facteur 2) : sinus / cosinus de la phase, PENTE du numéro de tour, carte de lignes.
Sert au tracé strict (tracer_strict.py). Le modèle reçoit l'image + la distance au centre d'enroulement.
    python inferer.py <dossier de travail> <poids.pt> <sortie> <nom1> [nom2 ...]      nom = cNNNN / rNNNNN (prep) ou paris4"""
import sys, os, json, math, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
D, POIDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; T = 512; dev = "cuda"; os.makedirs(OUT, exist_ok=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "entrainer.py"), encoding="utf-8").read(); exec(src[src.index("class Bloc"):src.index("def enr")])
s = torch.load(POIDS, map_location=dev); net = Traceur().to(dev); net.load_state_dict(s["net"]); net.eval(); print("poids du pas", s.get("pas"))
@torch.no_grad()
def brut(img, centre, pas=384, lot_=3):
    H, W = img.shape; x = torch.from_numpy(img.astype(np.float32) / 255.0); R_ = torch.from_numpy(np.hypot(np.arange(W, dtype=np.float32)[None, :] + 0.5 - centre[0], np.arange(H, dtype=np.float32)[:, None] + 0.5 - centre[1]))
    acc = torch.zeros(6, H, W); poids = torch.zeros(H, W); w = torch.hann_window(T, periodic=False).clamp(min=0.02); w2 = w[:, None] * w[None]
    ys = list(range(0, H - T + 1, pas)) + ([H - T] if (H - T) % pas else []); xs = list(range(0, W - T + 1, pas)) + ([W - T] if (W - T) % pas else [])
    P = [(y, x_) for y in ys for x_ in xs if img[y:y + T, x_:x_ + T].max() > 12]
    def entree(y, x_):
        r = R_[y:y + T, x_:x_ + T]; return torch.stack([x[y:y + T, x_:x_ + T], (r - r.mean()) / 256.0, r / 3000.0])
    for i in range(0, len(P), lot_):
        lot = P[i:i + lot_]; b = torch.stack([entree(y, x_) for y, x_ in lot]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16): o = net(b).float().cpu()
        for (y, x_), oo in zip(lot, o): acc[:, y:y + T, x_:x_ + T] += oo * w2; poids[y:y + T, x_:x_ + T] += w2
    return (acc / poids.clamp(min=1e-6)).numpy()
for n in sys.argv[4:]:
    if n == "paris4": img = cv2.imread(f"{D}/temoins/paris4_z9024.png", 0); ctr = json.load(open(f"{D}/temoins/paris4_centre.json"))["centre"]
    else: img = np.array(np.load(f"{D}/prep/{n}.npy", mmap_mode="r")[0]); ctr = json.load(open(f"{D}/prep/{n}.json"))["centre"]
    o = brut(img, ctr); h, w = img.shape[0] // 2, img.shape[1] // 2; r = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    # pente et densité : le réseau sort des tours par pixel x 10 ; enregistrées x 25 en entier 8 bits (donc valeur / 250 = tours par pixel de la pleine résolution)
    np.savez_compressed(f"{OUT}/{n}_brut.npz", sin=np.clip(r(o[1]) * 100, -127, 127).astype(np.int8), cos=np.clip(r(o[2]) * 100, -127, 127).astype(np.int8), ligne=(1 / (1 + np.exp(-r(o[0]))) * 255).astype(np.uint8),
                        gx=np.clip(r(o[3]) * 25, -127, 127).astype(np.int8), gy=np.clip(r(o[4]) * 25, -127, 127).astype(np.int8), dens=np.clip(r(np.exp(o[5])) * 25, 0, 255).astype(np.uint8), centre=np.array(ctr, np.float32), taille=np.array(img.shape), pas=s.get("pas")); print(n, img.shape, "->", (h, w), flush=True)
