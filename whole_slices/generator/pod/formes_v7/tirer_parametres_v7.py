"""24/11 · MODÈLE DE FORMES v6 (copie de tirer_parametres.py, non modifié) : tire les paramètres de spirale_champ_v6.py dans les plages MESURÉES sur les vraies coupes
(reel/mesurer.py sur 10 coupes de 0139, 0814, 0306B, 1667, 0332). Deux familles observées :
  « lache »  (1667, 0332, 0139 z5872)  : remplissage 0,26–0,40, écrasé 0–1 %, replis 3–14 /cm², pas de fissure ;
  « ecrase » (0306B, 0814, 0139 z12448): remplissage 0,51–0,67, écrasé 5–29 %, replis 18–110 /cm², fissures 0,7–3,5 % (courtes, à 55–70 % radiales).
Chaque tirage = une coupe différente (graine), mêmes lois pour toutes.
    python tirer_parametres_v6.py <nom> <famille lache|ecrase|mixte> <diametre_mm> <graine>  -> sorties/<nom>_prm.json"""
import sys, os, json, numpy as np
NOM, FAM, DMM, SEED = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
ONDF, CHF, FRF, TBF = (float(x) for x in (sys.argv[5:9] + ['1', '1', '1', '1'][len(sys.argv[5:9]):]))   # facteurs de calage (ondulations, chevrons, froissements, tourbillons)
r = np.random.default_rng(SEED + 7919)
if FAM == 'mixte': FAM = 'ecrase' if r.random() < 0.5 else 'lache'
U = lambda a, b: float(r.uniform(a, b))
ec = FAM == 'ecrase'
p = dict(nom=NOM, diametre_mm=DMM, graine=SEED, famille=FAM)
# espacement tour par tour (calé sur les histogrammes mesurés au même filtre de crêtes)
p['esp'] = dict(touche_px=U(3.6, 4.6), part_paquets=U(0.48, 0.62) if ec else U(0.42, 0.55), vide_med_px=U(10, 14) if ec else U(11, 17), vide_sigma=U(0.65, 0.85),
                paquet_tours=U(0.8, 1.6), paquet_arc_px=U(150, 400), vide_tours=U(0.5, 1.0), vide_arc_px=U(200, 500), tendance=U(0.0, 0.5) if ec else U(0.0, 0.2),
                lentilles_px=U(9, 16) if ec else U(7, 13), lentilles_tours=U(0.4, 0.9), lentilles_arc_px=U(30, 120), derive_max_px=U(8, 12) if ec else U(14, 22), derive_arc_px=U(180, 280))   # 24/11 : écrasé ±8–12 (v6_ec1/ec2 : 128–512 = 480/667), ±14–22 -> 1 082 (v6_ec3) ; lâche ±14–22 (v6_la3 : 502)
# écrasements non uniformes : un côté, des bandes à mi-rayon, parfois un pressage traversant
z = []
if ec:
    for _ in range(int(r.integers(1, 4))): z.append(dict(type='cote', u_rel=[U(0.4, 0.75), 1.0], angle_deg=U(-180, 180), demi_secteur_deg=U(30, 60), esp_px=U(2.8, 3.6)))
    for _ in range(int(r.integers(2, 6))):
        a = U(0.15, 0.8); z.append(dict(type='bande', u_rel=[a, min(1.0, a + U(0.15, 0.4))], angle_deg=U(-180, 180), demi_secteur_deg=U(15, 45), esp_px=U(2.8, 3.8)))
    if r.random() < 0.35: a = U(0.3, 0.6); z.append(dict(type='traversant', oppose=True, u_rel=[a, min(1.0, a + U(0.2, 0.4))], angle_deg=U(-180, 180), demi_secteur_deg=U(10, 25), esp_px=U(3.2, 4.5)))
else:
    for _ in range(int(r.integers(0, 3))): z.append(dict(type='cote', u_rel=[U(0.75, 0.9), 1.0], angle_deg=U(-180, 180), demi_secteur_deg=U(35, 65), esp_px=U(5.0, 6.5)))   # v6 : peu de tours tassés, pas collés
    if r.random() < 0.5: a = U(0.6, 0.85); z.append(dict(type='bande', u_rel=[a, min(1.0, a + U(0.08, 0.2))], angle_deg=U(-180, 180), demi_secteur_deg=U(5, 15), esp_px=U(3.5, 4.8)))
p['ecrasements'] = z
p['coeur'] = dict(angle_deg=U(-180, 180), tours=U(6, 18), lentille=U(0.5, 0.85) if ec else U(0.0, 0.5), trou_px=U(4, 14) if ec else U(10, 45))
p['ovale'] = dict(rapport=U(0.80, 0.97) if ec else U(0.75, 0.97), angle_deg=U(0, 180))
# plis en épingle : quelques tours repliés ensemble
npl = int(r.integers(12, 26)) if ec else int(r.integers(1, 5))
p['plis'] = [dict(u_rel=U(0.25, 0.92), angle_deg=U(-180, 180), largeur_px=U(20, 60), profondeur_px=U(40, 120), hauteur=U(0.8, 1.4), renversement=U(2.0, 4.0)) for _ in range(npl)]
# fissures : beaucoup de courtes (quelques tours), quelques coins depuis le bord
f = []
if ec:
    for _ in range(int(r.integers(1, 3))): a = U(0.55, 0.85); f.append(dict(angle_deg=U(-180, 180), r_rel=[a, 1.08], ouverture_px=U(12, 40), decalage_px=U(0, 14), sinuosite_px=U(5, 20), depuis_bord=True))
    for _ in range(int(r.integers(8, 25))): a = U(0.15, 0.9); f.append(dict(angle_deg=U(-180, 180), r_rel=[a, a + U(0.02, 0.06)], ouverture_px=U(6, 16), decalage_px=U(0, 6), sinuosite_px=U(1, 4), depuis_bord=False))
p['fissures'] = f
p['ond'] = [0, 0, 0, 0]                                                          # v6 : remplacé par ond6
A_ = float(r.uniform(0.9, 1.1)) * (0.6 if ec else 0.75) * ONDF                # calage du 24/11 : écrasé 0,6 (v6_ec1/ec2 : 32–128 = 517/551 ; 0,75 -> 936), lâche 0,75 (v6_la3 : 886)
p['ond6'] = [[8, A_ * U(0.06, 0.10)], [13, A_ * U(0.24, 0.32)], [45, A_ * U(0.01, 0.02)], [300, U(0.015, 0.03)]] if ec else             [[8, A_ * U(0.03, 0.06)], [13, A_ * U(0.22, 0.30)], [45, A_ * U(0.01, 0.02)], [300, U(0.015, 0.03)]]
# v6 · massifs écrasés (réel écrasé 5–29 % des fenêtres de 64 px ; lâche 0–1 %), chevrons et froissements DANS les massifs
part = U(0.12, 0.40) if ec else U(0.0, 0.015)
aire_m = part * np.pi * (DMM / 20) ** 2                                         # cm² de massif
p['massifs'] = dict(part=part, esp_px=[U(2.7, 3.0), U(3.2, 3.6)], cor_tours=U(6, 14), cor_arc_px=U(150, 350), u_min_rel=U(0.12, 0.25), u_max_rel=U(0.85, 0.95), amorti_ond=U(0.7, 0.9))
p['chevrons'] = dict(nombre=int(round(aire_m * U(16, 30) * CHF)), essais=3, longueur_onde_px=[20, 70], pente=[0.9, 2.0], long_px=[120, 400], epais_px=[30, 160])
p['froissements'] = dict(nombre=int(round(np.pi * (DMM / 20) ** 2 * (U(8, 16) if ec else U(0, 2)) * FRF)), essais=6, largeur_px=[5, 14], profondeur_px=[6, 18], hauteur=[0.8, 1.5], renversement=[1.5, 3.0])
p['sous_feuilles'] = dict(arc_px=U(40, 90), ecart_px=U(2.0, 3.5), seuil=U(0.3, 0.8), epaisseur_px=U(0.5, 1.2), franges_seuil=U(1.8, 2.4))   # échelles 10 / 24 / 90 / 300 px, calées sur le spectre de courbure mesuré
p['tourbillons'] = dict(nombre=int(round(np.pi * (DMM / 20) ** 2 * (U(10, 20) if ec else U(0, 1.5)) * TBF)), essais=6, rayon_px=[8, 28], angle_rad=[1.5, 3.0])
o = os.path.dirname(os.path.abspath(__file__)) + f'/sorties/{NOM}_prm.json'
json.dump(p, open(o, 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(o)
