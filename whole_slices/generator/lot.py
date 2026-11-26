"""25/11 · PRODUCTION D'UN LOT DE COUPES ENTIÈRES sur une machine louée (Linux, une ou plusieurs cartes NVIDIA). REPRENABLE : chaque étape est sautée si son fichier
existe déjà ; relancer la même commande après une coupure reprend là où ça s'est arrêté. Sur une AUTRE machine : déposer deja_faits.txt (numéros déjà rapatriés,
séparés par des espaces) à côté de lot.py, ils sont sautés.
  étape A (processeur, FORMES en parallèle) : forme (pod/geom_v7.py, plis sur carte) + cassures dans le tracé (casser_trace_entier.py ; la vérité reste continue)
  étape B (une coupe à la fois par carte)    : peinture (local_v7/peindre_masse.py, tracé abîmé dans les zones écrasées, STRIDE=192 = réglage certifié)
  étape C (processeur)                       : flou de mouvement, ton d'un vrai rouleau (PROFIL de 202 nombres, aucune vraie coupe ici), cailloux, bord
Chaque coupe tire sa famille (lâche / écrasée), son niveau de dégâts (0 intacte … 3 très abîmée) et son diamètre.
    python lot.py <dossier_sortie> <premier_numéro> <nombre> [cartes=0] [formes_en_parallèle=3] [diamètre_min=24] [diamètre_max=50]
Sortie : <dossier>/final/cN.png (image) + cN.npz (vérité : lig, ph8, U2, ded, lac0, cassure, jn, plages, meta) ; <dossier>/journal.txt (une ligne par coupe)."""
import sys, os, json, time, subprocess, threading, queue, shutil, numpy as np
OUT = os.path.abspath(sys.argv[1]); N0, NB = int(sys.argv[2]), int(sys.argv[3]); GPUS = (sys.argv[4] if len(sys.argv) > 4 else "0").split(","); KF = int(sys.argv[5]) if len(sys.argv) > 5 else 3
DMIN, DMAX = (float(sys.argv[6]) if len(sys.argv) > 6 else 24.0), (float(sys.argv[7]) if len(sys.argv) > 7 else 50.0)
R = os.path.dirname(os.path.abspath(__file__)); PY = [sys.executable, "-X", "utf8", "-W", "ignore"]; POIDS = os.environ.get("POIDS", R + "/ema_affine_p02000.pt"); PROFILS = json.load(open(R + "/profils_ton.json", encoding="utf-8"))
for d in ("geo", "cas", "jeu", "tmp", "final", "logs"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
DEJA = set(int(x) for x in open(R + '/deja_faits.txt').read().split()) if os.path.exists(R + '/deja_faits.txt') else set()   # numéros déjà rapatriés (reprise sur une autre machine)
verrou = threading.Lock()
def note(s):
    with verrou:
        print(time.strftime("%H:%M:%S"), s, flush=True); open(OUT + "/journal.txt", "a", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
def tirage(n):
    r = np.random.default_rng(n * 7919 + 11); u = lambda a, b: round(float(r.uniform(a, b)), 2)
    fam = os.environ.get("FAMILLE") or ("lache" if r.random() < 0.4 else "ecrase"); deg = int(os.environ.get("DEGATS") or r.choice([0, 1, 2, 3], p=[0.25, 0.30, 0.30, 0.15])); ec = fam == "ecrase"
    p = dict(FAM=fam, DEG=deg, DMM=round(float(r.triangular(DMIN, min(max(38.0, DMIN), DMAX), DMAX)), 1), FORCE=u(0.16, 0.24) if ec else u(0.08, 0.16), ESP=3.7 if ec else 3.4, EV=u(1.5, 2.1) if ec else u(1.3, 1.6),
             VS=u(0.7, 1.0) if ec else u(0.55, 0.75), MA=u(0.12, 0.35) if ec else 0.3, CAL=0.6 if ec else 0.4, PF=int(r.integers(2, 5)) if ec else int(r.integers(0, 2)), PENTE=u(0.45, 0.6) if ec else u(0.3, 0.45),
             AM=[0.0, u(0.15, 0.3), u(0.3, 0.45), u(0.4, 0.6)][deg] if ec else [0.0, 0.0, u(0.1, 0.25), u(0.2, 0.35)][deg], FISS=[0, 0, 12, 16][deg], COIN=[0, 0, int(r.choice([0, 60])), int(r.choice([60, 80, 100]))][deg],
             NFIS=[0, int(r.integers(0, 2)), int(r.integers(1, 4)), int(r.integers(3, 6))][deg], CX=[u(0.0, 0.3), u(0.2, 0.7), u(0.5, 1.1), u(0.8, 1.5)][deg],
             PLG=([u(0.0, 0.08), u(0.05, 0.18), u(0.12, 0.28), u(0.2, 0.35)][deg]) if ec else ([0.0, u(0.0, 0.06), u(0.04, 0.12), u(0.08, 0.18)][deg]))
    noms = [k for k, v in PROFILS.items() if v["famille"] == fam] or list(PROFILS); p["TON"] = noms[int(r.integers(len(noms)))]; return p
def lance(cmd, cwd, env, log):
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()})
    with open(log, "a", encoding="utf-8") as f: return subprocess.run(cmd, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
def forme(n, gpu):
    p = tirage(n); g = f"{OUT}/geo/c{n}.npz"; c = f"{OUT}/cas/c{n}/c{n}.npz"; log = f"{OUT}/logs/c{n}.txt"; t0 = time.time()
    if not os.path.exists(c):
        if not os.path.exists(g):
            env = dict(CUDA_VISIBLE_DEVICES=gpu, GPU=1, SANS_FISS_V6=1, PLIS_FRANCS=p["PF"], PENTE=p["PENTE"], COIN=p["COIN"], AMANDE=p["AM"], FISS=p["FISS"], DEC=14, CAILLOUX=p["CX"], TAILLE=1, FORCE=p["FORCE"], BORD=0.4, FAM=p["FAM"],
                       DMM=p["DMM"], ESP=p["ESP"], ESP_VIDE=p["EV"], VSIG=p["VS"], MASSIFS=p["MA"], PLANCHER=11.5, PLIS_MIN=8, DLT_MIN=4, ECH="[[0.5,1.0],[0.22,0.45]]", CALAGE=f"{p['CAL']} 0 0.5 0.3", SANS_COTES=1, OND300=0)
            if lance(PY + ["geom_v7.py", f"{OUT}/geo", str(n), str(n + 1)], R + "/pod", env, log) or not os.path.exists(g):
                env["GPU"] = 0; note(f"c{n} forme : échec sur carte (mémoire vidéo ?), reprise sur processeur")   # 25/11 : carte de 16 Go partagée avec la peinture
                if lance(PY + ["geom_v7.py", f"{OUT}/geo", str(n), str(n + 1)], R + "/pod", env, log) or not os.path.exists(g): note(f"c{n} FORME EN ÉCHEC (voir logs/c{n}.txt)"); return False
        os.makedirs(os.path.dirname(c), exist_ok=True)
        if lance(PY + ["casser_trace_entier.py", g, c + ".tmp.npz", str(n), str(p["NFIS"]), "50"], R, {}, log) or not os.path.exists(c + ".tmp.npz"): note(f"c{n} CASSURES EN ÉCHEC"); return False
        os.replace(c + ".tmp.npz", c)
    return round(time.time() - t0)
def peinture(n, gpu):
    p = tirage(n); j = f"{OUT}/jeu/c{n}.npz"; t0 = time.time()
    if not os.path.exists(j):
        env = dict(CUDA_VISIBLE_DEVICES=gpu, STRIDE=192, ABIME=1, PLAGES=p["PLG"], PLAGES_LACHE=p["PLG"], MASSE=0, DIL=60, GMIN=2.5, GMAX=2.5, SAUTER="aucune", SS=R + "/local_v7/gen")
        if lance(PY + ["peindre_masse.py", f"{OUT}/cas/c{n}", f"{OUT}/jeu", POIDS, "12", "12"], R + "/local_v7", env, f"{OUT}/logs/c{n}.txt") or not os.path.exists(j): note(f"c{n} PEINTURE EN ÉCHEC"); return False
    return round(time.time() - t0)
def finition(n):
    import cv2
    p = tirage(n); f = f"{OUT}/final/c{n}.png"; t = f"{OUT}/tmp/c{n}"; c = f"{OUT}/cas/c{n}/c{n}.npz"; log = f"{OUT}/logs/c{n}.txt"; t0 = time.time()
    if not os.path.exists(f):
        z = np.load(f"{OUT}/jeu/c{n}.npz"); cv2.imwrite(t + "_img.png", z["img"]); cv2.imwrite(t + "_plages.png", z["plages"]); plages = z["plages"]; meta = json.loads(str(z["meta"])); del z
        for cmd in (["flou_mouvement2.py", t + "_img.png", t + "_plages.png", t + "_flou.png", str(n)], ["ton_profil.py", t + "_flou.png", R + "/profils_ton.json", p["TON"], t + "_ton.png"],
                    ["cailloux.py", c, t + "_ton.png", t + "_cail.png", "40", str(n)], ["rogner_bord.py", c, t + "_cail.png", t + "_fini.png", "14", str(n)]):
            if lance(PY + cmd, R, {}, log): note(f"c{n} FINITION EN ÉCHEC à {cmd[0]}"); return False
        v = dict(np.load(c)); meta.update(lot=dict(tirage=p, ton=p["TON"])); v["meta"] = json.dumps(meta, default=float); v["plages"] = plages
        for k in ("vis0",): v.pop(k, None)
        np.savez_compressed(f"{OUT}/final/c{n}.tmp.npz", **v); os.replace(f"{OUT}/final/c{n}.tmp.npz", f"{OUT}/final/c{n}.npz"); os.replace(t + "_fini.png", f)
        if not os.environ.get("GARDER"):
            for x in (f"{OUT}/jeu/c{n}.npz", f"{OUT}/geo/c{n}.npz", c, t + "_img.png", t + "_plages.png", t + "_flou.png", t + "_ton.png", t + "_cail.png"):
                if os.path.exists(x): os.remove(x)
    return round(time.time() - t0)
qA, qB, qC = queue.Queue(), queue.Queue(maxsize=max(4, 3 * len(GPUS))), queue.Queue(); faits = [0]; T0 = time.time(); temps = {}
def travailA(i):
    while True:
        n = qA.get()
        if n is None: break
        if n in DEJA or os.path.exists(f"{OUT}/final/c{n}.png"): faits[0] += 1; continue
        s = forme(n, GPUS[i % len(GPUS)]) if not os.path.exists(f"{OUT}/jeu/c{n}.npz") else 0
        if s is not False: temps[n] = [s]; qB.put(n)
def travailB(gpu):
    while True:
        n = qB.get()
        if n is None: break
        s = peinture(n, gpu)
        if s is not False: temps[n].append(s); qC.put(n)
def travailC():
    while True:
        n = qC.get()
        if n is None: break
        try: s = finition(n)
        except Exception as e: note(f"c{n} FINITION EN ÉCHEC : {e!r}"); s = False
        if s is not False:
            faits[0] += 1; p = tirage(n); note(f"c{n} faite | {p['FAM']} dégâts {p['DEG']} {p['DMM']} mm | forme {temps[n][0]} s, peinture {temps[n][1]} s, finition {s} s | {faits[0]} / {NB} en {round((time.time() - T0) / 60)} min")
for n in range(N0, N0 + NB): qA.put(n)
A = [threading.Thread(target=travailA, args=(i,)) for i in range(KF)]; B = [threading.Thread(target=travailB, args=(g,)) for g in GPUS]; C = [threading.Thread(target=travailC) for _ in range(max(2, len(GPUS)))]   # 25/11 : sur un serveur à 11 cartes, 2 finitions à la fois bridaient tout
for t in A + B + C: t.start()
for _ in A: qA.put(None)
for t in A: t.join()
for _ in B: qB.put(None)
for t in B: t.join()
for _ in C: qC.put(None)
for t in C: t.join()
note(f"LOT TERMINÉ : {faits[0]} coupes sur {NB} en {round((time.time() - T0) / 60)} min")
