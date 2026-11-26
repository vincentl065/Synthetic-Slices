"""24/11 · GÉOMÉTRIE v6 (copie de geom.py, formes figées dans formes_v7/ ; coupes non valides gardées avec valide=false dans meta, à écarter à l entraînement).
Pour chaque coupe i : paramètres tirés (formes/tirer_parametres_v7.py, copie figée du 23/11 20 h de coupe_entiere) puis spirale v5
(formes/spirale_champ_v7.py, copie figée) exécutée par runpy pour récupérer ses champs internes :
  U  = enroulement continu (numéro de tour continu, sans coupure),  Fz = U − φ/2π (feuille = Fz entier),
  dedans (papyrus entre le cœur et le dernier tour), lacune (fissure : papyrus absent), visibles (lignes peintes).
Sortie geo/cNNN.npz recadrée sur le rouleau (+96 px, multiple de 128) :
  vis, ded, lac (pleine résolution, bits) ; lig (feuille vraie, bits) ; ph8 = frac(Fz) × 256 (pleine résolution) ; U2 = U à ×2 (float32).
    python geom.py <dossier_sortie> <i_debut> <i_fin> [pas=1]"""
import sys, os, json, runpy, subprocess, time, numpy as np
OUT = sys.argv[1]; I0, I1 = int(sys.argv[2]), int(sys.argv[3]); PAS = int(sys.argv[4]) if len(sys.argv) > 4 else 1; os.makedirs(OUT, exist_ok=True)
FD = os.path.dirname(os.path.abspath(__file__)) + '/formes_v7'; os.makedirs(FD + '/sorties', exist_ok=True)


def tirage(i):
    r = np.random.default_rng(50000 + i)
    fam = 'ecrase' if r.random() < 0.55 else 'lache'
    dmm = float(18 + 20 * r.beta(1.4, 2.0))                    # 18–38 mm (peinture : ~10 s par million de px sur 4090 à 8 pas)
    if os.environ.get('DMM'): dmm = float(os.environ['DMM'])
    if os.environ.get('FAM'): fam = os.environ['FAM']
    return fam, round(dmm, 1), 1000 + i


for i in range(I0, I1, PAS):
    nom = f'c{i:03d}'; f = f'{OUT}/{nom}.npz'
    if os.path.exists(f): continue
    t0 = time.time(); fam, dmm, seed = tirage(i)
    subprocess.run([sys.executable, '-X', 'utf8', f'{FD}/tirer_parametres_v7.py', nom, fam, str(dmm), str(seed)] + os.environ.get('CALAGE', '1 1 1 1').split(), check=True, capture_output=True)
    rp = np.random.default_rng(70000 + i); PTR = dict(force=float(os.environ.get('FORCE', rp.uniform(0.06, 0.30))), echelles=json.loads(os.environ['ECH']) if os.environ.get('ECH') else [[0.45, 1.0], [0.2, float(rp.uniform(0.1, 0.35))]])
    if os.environ.get('BORD'): PTR['bord'] = float(os.environ['BORD'])
    pj = f'{FD}/sorties/{nom}_prm.json'; prm = json.load(open(pj, encoding='utf-8')); prm['petrissage'] = PTR
    FE = float(os.environ.get('ESP', 1.0))                               # 24/11 : écarts ×ESP (v5/v6 : tours qui se touchent à ~4 px = 38 µm, massifs 3 px : plus fin qu'une feuille)
    if FE != 1.0:
        prm['esp']['touche_px'] *= FE; prm['esp']['vide_med_px'] *= float(os.environ.get('ESP_VIDE', 1.25)); prm['plancher_px'] = float(os.environ.get('PLANCHER', 5.0))
        for z_ in prm['ecrasements']: z_['esp_px'] *= FE
        if prm.get('massifs', {}).get('esp_px'): prm['massifs']['esp_px'] = [v_ * FE * 1.15 for v_ in prm['massifs']['esp_px']]
    if os.environ.get('DLT_MIN') and prm.get('sous_feuilles'):           # 24/11 : dédoublement dessiné seulement si les 2 couches sont à ≥ 2·DLT_MIN px (tuiles réelles : écart mini ≈ 4 px)
        prm['sous_feuilles']['dlt_min'] = float(os.environ['DLT_MIN']); prm['sous_feuilles']['ecart_px'] = max(prm['sous_feuilles']['ecart_px'], 1.6 * float(os.environ['DLT_MIN']))
    if os.environ.get('MASSIFS') and prm.get('massifs'): prm['massifs']['part'] *= float(os.environ['MASSIFS'])   # 24/11 : la densité doit venir du pétrissage, pas de massifs posés sur les anneaux
    if os.environ.get('PLIS_MIN'): prm['plis_esp_min'] = float(os.environ['PLIS_MIN'])
    if os.environ.get('TAILLE'): prm['taille_exacte'] = True
    if os.environ.get('VSIG'): prm['esp']['vide_sigma'] *= float(os.environ['VSIG'])        # étalement des vides (vrais tracés : 5e–95e centile 8–26 px seulement)
    if os.environ.get('AMANDE'):                                             # v7.3 : aplatissement en amande (h_rel) + fissure le long de la ligne des épingles
        ra = np.random.default_rng(81000 + i); prm['amande'] = dict(h_rel=float(os.environ['AMANDE']), angle_deg=float(ra.uniform(0, 180)))
        if float(os.environ.get('FISS', 0)) > 0: prm['amande']['fissure'] = dict(largeur_px=float(os.environ['FISS']), decalage_px=float(os.environ.get('DEC', 0)), corr_px=float(ra.uniform(70, 220)), biais=float(ra.uniform(-0.5, 0.2)), gauche=float(ra.uniform(0.3, 1.0)), droite=float(ra.uniform(0.3, 1.0)), debut_rel=float(ra.uniform(0.04, 0.2)), coin_px=float(os.environ.get('COIN', 0)), coin_cote=float(ra.choice([-1.0, 1.0])), coin_debut_rel=float(ra.uniform(0.3, 0.6)))
    if os.environ.get('CAILLOUX'):                                           # v7.3 : gros cailloux par cm² (relevé réel 0–1,8 / cm² pour ≥ 15 px ; on compte ici rayon médian 8 px)
        prm['cailloux'] = dict(nombre=int(round(float(os.environ['CAILLOUX']) * np.pi * (dmm / 20) ** 2)), rayon_med_px=8.0, rayon_sigma=0.5, rayon_max_px=26.0)
    if int(os.environ.get('PLIS_FRANCS', 0)) > 0:                           # v7.4 : plusieurs plis francs hors du centre (formes complexes des vrais rouleaux écrasés)
        rk = np.random.default_rng(83000 + i); prm['plis_francs'] = [dict(centre_rel=[float(rk.uniform(-0.75, 0.75)), float(rk.uniform(-0.75, 0.75))], angle_deg=float(rk.uniform(0, 180)), demi_largeur_rel=float(rk.uniform(0.15, 0.5)),
                                                                         longueur_rel=float(rk.uniform(0.6, 1.4)), pente=float(rk.choice([-1, 1]) * rk.uniform(0.35, 1.0) * float(os.environ.get('PENTE', 1.0)))) for _ in range(int(os.environ['PLIS_FRANCS']))]
    if os.environ.get('SANS_COTES'): prm['ecrasements'] = [z for z in prm['ecrasements'] if z['type'] != 'cote']
    if os.environ.get('SANS_FISS_V6'): prm['fissures'] = []                 # 25/11 : plus de fissure radiale rectiligne (Vincent : « pas juste des traits ») ; les fissures viennent de casser_trace_entier.py
    if os.environ.get('OND300'): prm['ond6'][-1][1] = float(os.environ['OND300'])
    json.dump(prm, open(pj, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    sys.argv = ['spirale_champ_v7.py', f'{FD}/sorties/{nom}_prm.json']
    g = runpy.run_path(f'{FD}/spirale_champ_v7.py', run_name='__main__')
    U, Fz, ded, lac, lig, vis = g['U'], g['Fz'], g['dedans'], g['lacune'] & g['dedans'], g['lignes'] > 0, g['papier'] > 0     # vis = « papier » (sous-feuilles, franges)
    info = {k: g['info'][k] for k in ('valide', 'tours', 'ecart_tours_px', 'controle', 'nb_ecrasements', 'nb_plis', 'nb_fissures', 'nb_chevrons', 'nb_froissements', 'nb_tourbillons', 'massifs_pct', 'amande', 'cailloux', 'plis_francs') if k in g['info']}
    jn = (g['J'] <= 0) & g['_okJ']                                          # 25/11 : où la carte se replie (à corriger : doit être vide pour un jeu d'entraînement)
    del g
    ys, xs = np.nonzero(ded); H, W = ded.shape
    y0, x0 = max(0, ys.min() - 96), max(0, xs.min() - 96); h = ((min(H, ys.max() + 97) - y0) // 128 + 1) * 128; w = ((min(W, xs.max() + 97) - x0) // 128 + 1) * 128
    def cr(a, fill):
        o = np.full((h, w), fill, a.dtype); s = a[y0:y0 + h, x0:x0 + w]; o[:s.shape[0], :s.shape[1]] = s; return o
    for c_ in info.get('cailloux', []): c_['x'] = round(c_['x'] - x0, 1); c_['y'] = round(c_['y'] - y0, 1)      # repère de la coupe recadrée
    U, Fz, ded, lac, lig, vis = cr(U.astype(np.float32), 0), cr(Fz.astype(np.float32), 0), cr(ded, False), cr(lac, False), cr(lig, False), cr(vis, False); jn = cr(jn, False)
    ph8 = (np.round((Fz - np.floor(Fz)) * 256).astype(np.int32) % 256).astype(np.uint8)       # phase pleine résolution (à ×2 : repliement dans les paquets serrés)
    U2 = U.reshape(h // 2, 2, w // 2, 2).mean((1, 3)).astype(np.float32)
    meta = dict(nom=nom, famille=fam, diametre_mm=dmm, graine=seed, taille=[int(h), int(w)], forme='v7 (v6 + petrissage, 24/11)', petrissage=PTR, **info, secondes=round(time.time() - t0, 1))
    try:
        import psutil; meta['memoire_pic_go'] = round(getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss) / 2 ** 30, 1)   # pic de mémoire vive du calcul (pour choisir la taille)
    except Exception: pass
    np.savez_compressed(f + '.tmp.npz', vis=np.packbits(vis, axis=-1), ded=np.packbits(ded, axis=-1), lig=np.packbits(lig, axis=-1), ph8=ph8,
                        U2=U2, lac=np.packbits(lac, axis=-1), jn=np.packbits(jn, axis=-1), meta=json.dumps(meta, default=float))
    os.replace(f + '.tmp.npz', f)
    for e in ('.npz', '.json'):
        p = f'{FD}/sorties/{nom}{e}'
        if os.path.exists(p): os.remove(p)
    print(json.dumps(meta, ensure_ascii=False, default=float), flush=True)
