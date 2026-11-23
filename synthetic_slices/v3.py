"""v0.2 layouts and post-painting: the recipe behind the "S5" training set of the benchmark.

One tile = (1) a layout drawn at random among several kinds, (2) the diffusion model paints the slice around it (guidance 1.6 for
fold and hard-case layouts, 1.0 for plain ones), (3) a few phenomena are painted AFTER the model, without moving any line.

Kinds (shares of the S5 recipe, RECIPE_S5):
  - hard cases, 25 %: crack (jagged break, sheets cut and offset), mush (crushed sheets), delamination;
  - folds, 25 %: hairpin, tight U, tight bend, accordion (30 % of them "compact": sheets packed near the minimum gap);
  - plain stacks, 50 % (layout.py, as in v0.1), of which 8 % become a scroll edge with frayed last sheets.
  Extras: 25 % of non-hard tiles get sheets pressed into contact; mush tiles are painted "crushed" over the whole tile and 15 % of the
  others get a crushed patch; 25 % get bright mineral inclusions, 25 % get debris in the air.
Every line keeps its exact sheet id; two different sheets never touch (>= 3 px apart).

    lay = random_layout(seed, k)                  # dict: sheet_id, lines, dist, dist_cond, crack, air, crushed, kind, guidance
    img = paint_after(img, lay, seed, k)          # img: uint8 (H, W) from the diffusion model -> uint8 with crack / crush / extras
"""
import numpy as np
from scipy import ndimage as ndi
from . import layout as S4, folds as SP, hard_cases as SV, cracks as S3, phenomena as PH

RECIPE_S5 = dict(folds=0.25, hard=0.25, edge=0.08, inclusions=0.25, debris=0.25, contact=0.25, crushed_patch=0.15)
RECIPE_V01 = dict(folds=0.0, hard=0.0, edge=0.0, inclusions=0.0, debris=0.0, contact=0.0, crushed_patch=0.0)

KIND_EN = {"ordinaire": "plain", "epingle": "hairpin", "U_serre": "tight_U", "virage_serre": "tight_bend", "accordeon": "accordion",
           "tourbillon": "swirl", "fissure2": "crack", "fissure": "crack_v1", "mush": "mush", "delamination": "delamination",
           "bord": "scroll_edge"}


def kind_en(g):
    """French internal kind name ("epingle_compact+contact+ecrase") -> English ("hairpin_compact+contact+crushed")"""
    parts = g.split("+"); base = parts[0]; comp = base.endswith("_compact"); base = base.replace("_compact", "")
    out = KIND_EN.get(base, base) + ("_compact" if comp else "")
    for q in parts[1:]: out += "+" + {"contact": "contact", "ecrase": "crushed"}.get(q, q)
    return out


def random_layout(seed, k, size=256, margin=48, recipe=RECIPE_S5):
    """layout of tile k of a run with this seed (same draws as the benchmark generator for size 256 and RECIPE_S5)"""
    for essai in range(20):
        try: return _layout(seed, k, essai, (size, size) if np.isscalar(size) else tuple(size), margin, recipe)
        except (IndexError, ValueError): continue
    raise RuntimeError(f"layout {k}: no valid draw after 20 tries")


def _layout(G0, k, essai, N, MG, R):
    PC, PV2, PB, PCONT, PECR = R["folds"], R["hard"], R["edge"], R["contact"], R["crushed_patch"]
    rg = np.random.default_rng([G0, k, essai]); V = None; u = rg.random()
    if u < PV2:
        gk = SV.GENRES_V2[int(rg.integers(len(SV.GENRES_V2)))]
        if gk == "fissure":
            while V is None: r = S4.reglages(rg); V = S3.volume_fissure2(r, rg, N, MG)
            gk = "fissure2"
        else:
            while V is None: r = S4.reglages(rg); V = SV.volume_v2(r, rg, N, gk, MG)
        gen, gui = gk, 1.6
    elif u < PV2 + PC:
        gk = SP.GENRES[int(rg.integers(len(SP.GENRES)))]; cp = bool(rg.random() < 0.3)
        while V is None: r = S4.reglages(rg); V = SP.volume_pli(r, rg, N, gk, MG, compact=cp)
        gen, gui = gk + ("_compact" if cp else ""), 1.6
    else:
        r = S4.reglages(rg); V = S4.volume(r, rg, 1, N, MG); gen, gui = "ordinaire", 1.0
    idm = np.asarray(V["axe_id"]); idm = idm[0] if idm.ndim == 3 else idm
    fe = np.asarray(V["fente"])[0] if "fente" in V else np.zeros(idm.shape, np.uint8)
    air = np.zeros(idm.shape, bool)
    if u >= PV2 and rg.random() < PCONT / max(1e-9, 1 - PV2):
        idm2, na = PH.contact(idm, rg)
        if na: idm = idm2; gen = gen + "+contact"
    if u >= PV2 + PC and rg.random() < PB / max(1e-9, 1 - PV2 - PC):
        idm2, air = PH.bord(idm, rg)
        if air.mean() > 0.08 and (idm2 > 0).sum() > 200: idm = idm2; gen = "bord"
        else: air = np.zeros(idm.shape, bool)
    sk = idm > 0
    ecr = 1 if gen.startswith("mush") else (2 if (gen != "bord" and rg.random() < PECR) else 0)
    if ecr == 2: gen = gen + "+ecrase"
    d = ndi.distance_transform_edt(~sk)
    return dict(sheet_id=idm.astype(np.uint16), lines=sk.astype(np.uint8), dist=np.clip(np.round(d), 0, 255).astype(np.uint8),
                dist_cond=(np.clip(np.round(d * 4), 0, 255) / 4).astype(np.float32),
                crack=(fe > 0).astype(np.uint8), air=air, crushed=int(ecr), kind=kind_en(gen), guidance=float(gui))


def paint_after(img, lay, seed, k, recipe=RECIPE_S5):
    """painting steps after the diffusion model, in the benchmark order: crushed look, crack, then inclusions / debris / edge air.
    Returns (image, dict(inclusions=n, debris=n))."""
    im = img
    L = lay["lines"] > 0
    if lay["crushed"]:
        im, _ = S3.peindre_ecrase(im, L, np.random.default_rng([seed, int(k), 5]), plaque=bool(lay["crushed"] == 2))
    if lay["crack"].any():
        im, _ = S3.peindre_fente(im, lay["crack"] > 0, L, np.random.default_rng([seed, int(k), 7]))
    rg = np.random.default_rng([seed, int(k), 11]); a, b = rg.random() < recipe["inclusions"], rg.random() < recipe["debris"]; inf = {}
    if a or b or lay["air"].any():
        im, inf = PH.peindre(im, L, lay["air"], rg, inclusions=a, debris=b)
    return im, dict(inclusions=int(inf.get("inclusions", 0)), debris=int(inf.get("debris", 0)))
