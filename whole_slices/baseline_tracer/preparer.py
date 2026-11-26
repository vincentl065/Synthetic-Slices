"""25/11 · PRÉPARE les coupes pour l'entraînement du traceur : télécharge du dépôt Hugging Face privé ce qui manque, puis convertit chaque coupe en tableaux
lisibles par morceaux (image, lignes de vérité, phase entre feuilles, intérieur du rouleau). Tourne EN BOUCLE : les nouvelles coupes arrivent au fil de l'eau.
    python preparer.py <dossier de travail> [pause_s=120]        jeton facultatif dans HF_TOKEN ; s'arrête quand <dossier>/STOP existe"""
import sys, os, time, json, numpy as np, cv2
from huggingface_hub import HfApi, hf_hub_download
D = sys.argv[1]; PAUSE = int(sys.argv[2]) if len(sys.argv) > 2 else 120; REPO = "vincentl065/synthetic-scroll-slices"; TOK = os.environ.get("HF_TOKEN") or None; api = HfApi(token=TOK)
os.makedirs(D + "/prep", exist_ok=True); os.makedirs(D + "/brut", exist_ok=True)
def prepare(n):
    p = hf_hub_download(REPO, f"coupes/{n}.png", repo_type="dataset", token=TOK, local_dir=D + "/brut"); q = hf_hub_download(REPO, f"coupes/{n}.npz", repo_type="dataset", token=TOK, local_dir=D + "/brut")
    a = cv2.imread(p, 0); z = np.load(q); H, W = a.shape; bits = lambda k: (np.unpackbits(z[k], axis=-1)[:, :W] > 0).astype(np.uint8)
    lig, ded, cas = bits("lig"), bits("ded"), bits("cassure"); ph = z["ph8"]; assert ph.shape == a.shape and lig.shape == a.shape, (n, a.shape, ph.shape, lig.shape)
    m = json.loads(str(z["meta"])); t = m.get("lot", {}).get("tirage", {})
    np.save(f"{D}/prep/{n}_tmp.npy", np.stack([a, lig * 255, ph, ded * 255, cas * 255])); os.replace(f"{D}/prep/{n}_tmp.npy", f"{D}/prep/{n}.npy")
    json.dump(dict(nom=n, taille=[H, W], famille=t.get("FAM"), degats=t.get("DEG"), diametre=t.get("DMM")), open(f"{D}/prep/{n}.json", "w")); os.remove(p); os.remove(q)
while not os.path.exists(D + "/STOP"):
    try: F = set(s.rfilename for s in api.dataset_info(REPO).siblings)
    except Exception as e: print("liste du dépôt en échec :", repr(e)[:120], flush=True); time.sleep(60); continue
    noms = sorted((f[7:-4] for f in F if f.startswith("coupes/") and f.endswith(".png") and f[:-4] + ".npz" in F), key=lambda s: int(s[1:])); PRIO = os.environ.get("TEMOINS", "").split(","); neufs = sorted((n for n in noms if not os.path.exists(f"{D}/prep/{n}.npy")), key=lambda n: (n not in PRIO, int(n[1:])))   # les témoins d'abord
    for n in neufs:
        try: prepare(n)
        except Exception as e: print(n, "en échec :", repr(e)[:160], flush=True)
    if neufs: print(time.strftime("%H:%M:%S"), f"{len(neufs)} coupes préparées | total {len([x for x in os.listdir(D + '/prep') if x.endswith('.npy')])}", flush=True)
    time.sleep(PAUSE)
