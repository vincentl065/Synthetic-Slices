"""26/11 · Ajoute à chaque coupe préparée la position de son CENTRE D'ENROULEMENT (lue dans la vérité U2 du dépôt Hugging Face) : <prep>/cN.json reçoit "centre": [x, y].
Tourne en boucle (les nouvelles coupes arrivent). Le réseau reçoit alors, en plus de l'image, la distance à ce centre : il sait de quel côté est l'extérieur.
    python centres.py <dossier de travail> [pause_s=120]"""
import sys, os, json, time, glob, numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
D = sys.argv[1]; PAUSE = int(sys.argv[2]) if len(sys.argv) > 2 else 120; REPO = "vincentl065/synthetic-scroll-slices"; TOK = os.environ.get("HF_TOKEN") or None; os.makedirs(D + "/brut2", exist_ok=True)
def centre(j):
    try:
        m = json.load(open(j)); n = m["nom"]
        if "centre" in m: return 0
        q = hf_hub_download(REPO, f"coupes/{n}.npz", repo_type="dataset", token=TOK, local_dir=D + "/brut2"); U = np.load(q)["U2"]; H, W = m["taille"]
        v = cv2.GaussianBlur(np.where(U >= 0, U, 1e6).astype(np.float32), (0, 0), 3); y, x = np.unravel_index(np.argmin(v), U.shape); m["centre"] = [float(x) * W / U.shape[1], float(y) * H / U.shape[0]]
        json.dump(m, open(j + ".tmp", "w")); os.replace(j + ".tmp", j); os.remove(q); return 1
    except Exception as e:
        print(j, "en échec :", repr(e)[:120], flush=True); return 0
while not os.path.exists(D + "/STOP"):
    L = [j for j in sorted(glob.glob(D + "/prep/c*.json")) if '"centre"' not in open(j).read()]
    if L:
        with ThreadPoolExecutor(8) as ex: n = sum(ex.map(centre, L))
        print(time.strftime("%H:%M:%S"), n, "centres ajoutés | restants", len(L) - n, flush=True)
    time.sleep(PAUSE)
