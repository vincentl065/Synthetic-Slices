"""24/11 · VERSION GUIDÉE (guidage tiré par coupe dans [GMIN, GMAX], défaut 2–4) : sans guidage l image ne suivait pas le tracé
(corrélation image / phase 0,07 ; 0,20 à guidage 2 ; 0,30 à guidage 4). Ordre : c000, c001 puis les coupes d entraînement de la plus petite à la plus grande ;
c002 / c003 sautées (SAUTER). Arrêt si STOP ou si l heure UNIX dépasse FIN_PEINTURE.
23/11 · traceur de coupe entière — PEINTURE du jeu (GPU, pod). Générateur affiné v0.3 (ema_affine_p02000.pt : jamais vu Paris 4,
bord appris ; canal 0 = masque du rouleau). Une coupe = un style tiré (0–8 ou « inconnu », JAMAIS l'indice Paris4), canevas sans raccord.
Masque à cellules de 16 px comme les vrais volumes « masked », trou central rempli (correctif du 23/11), hors masque = 0 franc.
Tourne en boucle sur geo/*.npz tant que le fichier STOP n'existe pas ; écrit jeu/cNNN.npz (image + étiquettes de geo).
    python peindre_jeu.py <geo> <jeu> <poids.pt> [pas=12] [fenetres=32]"""
import sys, os, glob, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SS", os.path.dirname(os.path.abspath(__file__)) + "/gen"))
import synthetic_slices as S
from synthetic_slices.model import UNet, DMAX, NS
from synthetic_slices import sampler_rapide as R
STRIDE = int(os.environ.get('STRIDE', 0))   # 25/11 : STRIDE=192 = peintre rapide certifié de qualité identique (equivalence2.py, 24/11) ; 0 = sampler.py d'origine (stride 128)
from scipy import ndimage as ndi
GEO, JEU, P = sys.argv[1:4]; PAS = int(sys.argv[4]) if len(sys.argv) > 4 else 12; NW = int(sys.argv[5]) if len(sys.argv) > 5 else 32
os.makedirs(JEU, exist_ok=True); dev = "cuda"; dt = torch.float16 if getattr(torch.version, "hip", None) else torch.bfloat16   # carte AMD : fp16
net = UNet().to(dev).eval(); net.load_state_dict(torch.load(P, map_location=dev))
STYLES = list(range(9)) + [NS]
GMIN, GMAX = float(os.environ.get('GMIN', 2)), float(os.environ.get('GMAX', 4)); SAUTER = set(os.environ.get('SAUTER', 'c002,c003').split(',')); FINP = float(os.environ.get('FIN_PEINTURE', 1e18))


def cond(m, l):
    d = ndi.distance_transform_edt(~l); d = np.clip(np.round(d * 4), 0, 255) / 4
    return torch.stack([torch.tensor(m.astype(np.float32)), torch.tensor(l.astype(np.float32)) * 2 - 1, torch.tensor(d, dtype=torch.float32).clamp(max=DMAX) / (DMAX / 2) - 1])


while not os.path.exists(JEU + "/STOP"):
    faits = {os.path.basename(f) for f in glob.glob(JEU + "/c*.npz")}
    todo = [f for f in sorted(glob.glob(GEO + "/c*.npz")) if os.path.basename(f) not in faits and not f.endswith(".tmp.npz")]
    todo = [f for f in todo if os.path.basename(f)[:4] not in SAUTER]
    if not todo or time.time() > FINP: break
    tete = [f for f in todo if os.path.basename(f)[:4] in ('c000', 'c001')]
    f = tete[0] if tete else min(todo, key=os.path.getsize); z = dict(np.load(f)); meta = json.loads(str(z["meta"])); t0 = time.time()
    H, W = meta["taille"]; L = np.unpackbits(z["vis"], axis=-1)[:, :W] > 0; DE = np.unpackbits(z["ded"], axis=-1)[:, :W] > 0
    i = int(meta["nom"][1:]); r = np.random.default_rng(777 + i); sty = int(STYLES[r.integers(len(STYLES))])
    dd = ndi.binary_fill_holes(ndi.binary_dilation(DE, iterations=int(os.environ.get('DIL', 20))))   # 24/11 : DIL=60 -> marge grise autour du papyrus comme les vraies coupes (le bord en escalier ne mord plus les feuilles)
    m = np.repeat(np.repeat(dd.reshape(H // 16, 16, W // 16, 16).any((1, 3)), 16, 0), 16, 1)
    PLG = np.zeros((H, W), np.float32)
    if os.environ.get('ABIME'):                                             # 24/11 : tracé abîmé pour le peintre (trace_abime.py) ; la vérité ne change pas
        import trace_abime; L, PLG = trace_abime.abimer(L, DE, 52000 + i, part_plages=float(os.environ.get('PLAGES', 0.4)) if meta.get('famille') == 'ecrase' else float(os.environ.get('PLAGES_LACHE', 0.1)), bouts_plage=float(os.environ.get('BOUTS', 10)), coup_plage=float(os.environ.get('COUP', 0.22)))
    C = cond(m, L)[None].to(dev); gen = torch.Generator(device=dev).manual_seed(4000 + i)
    gui = float(np.random.default_rng(9000 + i).uniform(GMIN, GMAX))
    # 24/11 · MASSES ÉCRASÉES (test à l'aveugle de Vincent : il ne juge « vrai » que les zones tissées / écrasées) : plages tirées d'un champ lisse, couvrant
    # MASSE (part de la surface du rouleau) ; dedans le guidage tombe à GMASSE (≈ 1 : le générateur peint son « tapis de fibres » sans suivre feuille par feuille).
    PM = float(os.environ.get('MASSE', 0)); masse = np.zeros((H, W), np.float32)
    if PM > 0:
        import cv2
        rm = np.random.default_rng(31000 + i); sg = float(os.environ.get('MASSE_PX', 260)); fq = 16
        g = cv2.GaussianBlur(rm.normal(0, 1, (H // fq + 2, W // fq + 2)).astype(np.float32), (0, 0), sg / fq); g = cv2.GaussianBlur(cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 2 * fq)
        seuil = np.quantile(g[DE][::7], 1 - PM); masse = np.clip((g - seuil) / (0.25 * g[DE].std()) + 0.5, 0, 1).astype(np.float32) * DE
        masse = cv2.GaussianBlur(masse, (0, 0), 12)
        gui = torch.tensor(gui + (float(os.environ.get('GMASSE', 1.0)) - gui) * masse, device=dev)[None, None]
    out = R.sample(net, C, torch.tensor([sty], device=dev).long(), H, W, gen, dev, dt, NW, PAS, guidance=gui, stride=STRIDE, solver='heun') if STRIDE else S.sample(net, C, torch.tensor([sty], device=dev).long(), H, W, gen, dev, dt, NW, PAS, guidance=gui)
    gui = float(os.environ.get('GMIN', 2)) if PM > 0 else gui; z['masse'] = (masse * 255).astype(np.uint8); z['plages'] = (PLG * 255).astype(np.uint8); z['trace_peintre'] = np.packbits(L, axis=-1)
    img = S.to_uint8(out)[0, 0].cpu().numpy(); img = np.where(m, np.maximum(img, 1), 0).astype(np.uint8)
    meta.update(style=sty, pas=PAS, guidage=round(gui, 2), masse=PM, guidage_masse=float(os.environ.get('GMASSE', 1.0)), peinture_s=round(time.time() - t0, 1), s_par_mpx=round((time.time() - t0) / (H * W / 1e6), 1))
    z["img"] = img; z["meta"] = json.dumps(meta)
    np.savez(JEU + "/" + os.path.basename(f)[:-4] + ".tmp.npz", **z); os.replace(JEU + "/" + os.path.basename(f)[:-4] + ".tmp.npz", JEU + "/" + os.path.basename(f))
    print(json.dumps(meta, ensure_ascii=False), flush=True)
