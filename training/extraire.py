"""Build the REAL training pairs from the public bucket s3://vesuvius-challenge-open-data (no login needed):
  - 9 scrolls at ~9 um: level-0 CT + level-0 m7 surface prediction (APPROXIMATE layouts), 384^3 blocks where m7 is dense,
    3 blocks per scroll held out for testing; 256 x 256 tiles along z AND along the scroll axis; sheet axes = 1-px skeleton;
  - Paris 4 (harmonised instance labels, EXACT layouts), 8 held-out cubes.
Run on a machine with a good connection (we used a cloud GPU pod):  nice -n 19 python extraire.py [blocks_per_scroll=30] [slices_per_axis=20]"""
import sys, os, glob, json, time, numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_otsu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import ROULEAUX, STYLES, DONNEES, journal
T, BL = (256, 384)
_Z = {}

def ouvrir(nom):
    if nom not in _Z:
        import zarr, s3fs
        fs = s3fs.S3FileSystem(anon=True)
        vid = ROULEAUX[nom].split('-')[0]
        v = zarr.open_array(s3fs.S3Map(f'vesuvius-challenge-open-data/{nom}/volumes/{ROULEAUX[nom]}-masked.zarr/0', s3=fs), mode='r')
        m = zarr.open_group(s3fs.S3Map(f'vesuvius-challenge-open-data/{nom}/representations/predictions/surfaces/{vid}-surface-20260413222639-surface-m7-L0-th0.2.zarr', s3=fs), mode='r')
        _Z[nom] = (v, m)
    return _Z[nom]

def note_m7(m):
    sk = skeletonize(m)
    ep = m.sum() / max(1, sk.sum())
    g = ndi.gaussian_filter(m.astype(np.float32), 1.5)
    gy, gx = np.gradient(g)
    jxx, jyy, jxy = (ndi.gaussian_filter(a, 12) for a in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-09)
    w = jxx + jyy
    return (float(ep), float((coh * w).sum() / (w.sum() + 1e-09)))

def papyrus(ct):
    g = ndi.gaussian_filter(ct.astype(np.float32), 1.0)
    m = g > threshold_otsu(g)
    m = ndi.binary_opening(m, structure=ndi.generate_binary_structure(2, 1))
    return remove_small_objects(m, 40)

def fiche(ct, sk, m7=None):
    sk = remove_small_objects(sk.astype(bool), 12, connectivity=2)
    d = ndi.distance_transform_edt(~sk) if sk.any() else np.full(sk.shape, 99.0)
    pap = papyrus(ct)
    pf = float(pap.mean())
    nonlab = float((pap & (d > 10)).sum() / max(1, pap.sum()))
    faux = float((sk & ~ndi.binary_dilation(pap, iterations=2)).sum() / max(1, sk.sum()))
    ep, coh = note_m7(m7) if m7 is not None and m7.any() else (0.0, 1.0)
    return dict(ct=ct, lig=np.packbits(sk, axis=1), dist=np.clip(np.round(d * 4), 0, 255).astype(np.uint8), q=np.array([ep, coh, pf, nonlab, faux, float(sk.mean())], np.float32))

def bloc_m7(arg):
    nom, k, (bz, by, bx), na, graine, test = arg
    rng = np.random.default_rng(graine)
    v, g = ouvrir(nom)
    t0 = time.time()
    try:
        ct = np.asarray(v[bz:bz + BL, by:by + BL, bx:bx + BL])
        m7 = np.asarray(g['0'][bz:bz + BL, by:by + BL, bx:bx + BL]) > 0
    except Exception as e:
        return (nom, k, [], f'erreur {e}')
    if ct.shape != (BL, BL, BL) or (ct > 0).mean() < 0.9:
        return (nom, k, [], 'hors rouleau')
    res = []
    for axe in range(3):
        for j in range(na):
            i = int((j + rng.uniform(0.1, 0.9)) * BL / na)
            a, c = (int(x) for x in rng.integers(0, BL - T + 1, 2))
            if axe == 0:
                t, m = (ct[i, a:a + T, c:c + T], m7[i, a:a + T, c:c + T])
            elif axe == 1:
                t, m = (ct[a:a + T, i, c:c + T], m7[a:a + T, i, c:c + T])
            else:
                t, m = (ct[a:a + T, c:c + T, i], m7[a:a + T, c:c + T, i])
            t = np.ascontiguousarray(t)
            m = np.ascontiguousarray(m)
            if (t > 0).mean() < 0.98 or m.mean() < 0.01:
                continue
            f = fiche(t, skeletonize(m), m)
            f.update(axe=axe, zyx=(bz, by, bx, axe, i, a, c))
            res.append(f)
    return (nom, k, res, f'{len(res)} tuiles, {time.time() - t0:.0f} s')

def cube_paris4(arg):
    dossier, k, lo, hi, test = arg
    import nrrd
    n = os.path.basename(dossier)
    v, _ = nrrd.read(f'{dossier}/{n}_volume.nrrd', index_order='C')
    lab, _ = nrrd.read(f'{dossier}/{n}_mask.nrrd', index_order='C')
    v8 = np.clip((v.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    res = []
    for axe in range(3):
        for i in range(8, 256, 16):
            t = np.take(v8, i, axis=axe)
            l = np.take(lab, i, axis=axe)
            if (l > 0).mean() < 0.08:
                continue
            P = np.pad(l, 1, mode='edge')
            bord = np.zeros(l.shape, bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        nb = P[1 + dy:1 + dy + l.shape[0], 1 + dx:1 + dx + l.shape[1]]
                        bord |= (nb != l) & (nb > 0) & (l > 0)
            sk = skeletonize((l > 0) & ~bord)
            f = fiche(np.ascontiguousarray(t), sk)
            f.update(axe=0 if axe == 0 else 1, zyx=tuple((int(x) for x in n.split('_'))) + (axe, i, 0, 0))
            res.append(f)
    return ('Paris4', k, res, f'{len(res)} tuiles')

def main():
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    na = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    SUF = sys.argv[3] if len(sys.argv) > 3 else ''
    os.makedirs(DONNEES, exist_ok=True)
    rng = np.random.default_rng(1911 + len(SUF) * 77)
    t0 = time.time()
    taches = []
    for s, nom in enumerate(ROULEAUX):
        if os.path.exists(f'{DONNEES}/tuiles_{s:02d}_{nom}{SUF}.npz'):
            continue
        v, g = ouvrir(nom)
        L5 = np.asarray(g['5'][:]).astype(np.float32)
        f = BL // 32
        nz, ny, nx = (x // f for x in L5.shape)
        sc = L5[:nz * f, :ny * f, :nx * f].reshape(nz, f, ny, f, nx, f).mean((1, 3, 5))
        zz = np.arange(nz)[:, None, None]
        ok = (sc >= 0.5 * np.percentile(sc[sc > 0], 90)) & (zz >= 0.08 * nz) & (zz <= 0.92 * nz)
        if SUF:
            import zarr, s3fs
            V5 = np.asarray(zarr.open_array(s3fs.S3Map(f'vesuvius-challenge-open-data/{nom}/volumes/{ROULEAUX[nom]}-masked.zarr/5', s3=s3fs.S3FileSystem(anon=True)), mode='r')[:])
            dd = (V5[:nz * f, :ny * f, :nx * f] > 0).reshape(nz, f, ny, f, nx, f).mean((1, 3, 5))
            ok &= dd >= 0.97
            z1 = np.load(f'{DONNEES}/tuiles_{s:02d}_{nom}.npz')
            b1 = z1['zyx'][:, :3] // BL
            for a, b, c in {tuple(x) for x in b1[z1['test']]}:
                ok[max(0, a - 1):a + 2, max(0, b - 1):b + 2, max(0, c - 1):c + 2] = False
            for a, b, c in {tuple(x) for x in b1}:
                ok[a, b, c] = False
        cand = np.argwhere(ok)
        ch = cand[rng.choice(len(cand), min(nb, len(cand)), replace=False)]
        test = set() if SUF else set(rng.choice(len(ch), 3, replace=False).tolist())
        print(f"{nom} : {len(cand)} blocs candidats, {len(ch)} pris (3 d'examen)", flush=True)
        for k, (a, b, c) in enumerate(ch):
            taches.append((nom, k, (int(a) * BL, int(b) * BL, int(c) * BL), na, int(rng.integers(1 << 31)), k in test))
    cubes = sorted(glob.glob(f'{DONNEES}/paris4/instance-labels-harmonized/*_*_*'))
    tp4 = []
    if cubes and (not os.path.exists(f'{DONNEES}/tuiles_{len(ROULEAUX):02d}_Paris4.npz')):
        import nrrd
        pc = [np.percentile(nrrd.read(f'{d}/{os.path.basename(d)}_volume.nrrd')[0][::4, ::4, ::4], [0.5, 99.5]) for d in cubes[::max(1, len(cubes) // 6)][:6]]
        lo, hi = np.median(pc, 0)
        test = set(rng.choice(len(cubes), 8, replace=False).tolist())
        tp4 = [(d, k, float(lo), float(hi), k in test) for k, d in enumerate(cubes)]
    tests = {(t[0], t[1]): t[-1] for t in taches}
    tests.update({('Paris4', t[1]): t[-1] for t in tp4})
    restant = {nom: sum((1 for t in taches if t[0] == nom)) for nom in ROULEAUX}
    restant['Paris4'] = len(tp4)
    acc = {n: [] for n in STYLES}
    _Z.clear()
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(2) as pool:
        for nom, k, res, msg in pool.imap(lambda_dispatch, [('m7', t) for t in taches] + [('p4', t) for t in tp4]):
            s = STYLES.index(nom)
            for f in res:
                f.update(style=s, test=tests[nom, k], bloc=s * 1000 + k + (500 if SUF else 0))
            acc[nom] += res
            restant[nom] -= 1
            print(time.strftime('%H:%M:%S'), nom, k, msg, f'· {time.time() - t0:.0f} s', flush=True)
            if restant[nom] == 0 and acc[nom]:
                A = acc[nom]
                np.savez_compressed(f"{DONNEES}/tuiles_{s:02d}_{nom}{(SUF if nom != 'Paris4' else '')}.npz", ct=np.stack([f['ct'] for f in A]), lig=np.stack([f['lig'] for f in A]), dist=np.stack([f['dist'] for f in A]), q=np.stack([f['q'] for f in A]), style=np.array([f['style'] for f in A], np.int8), axe=np.array([min(1, f['axe']) for f in A], np.int8), test=np.array([f['test'] for f in A]), bloc=np.array([f['bloc'] for f in A], np.int32), zyx=np.array([f['zyx'] for f in A], np.int32))
                print(f"ÉCRIT {nom} : {len(A)} tuiles ({sum((f['test'] for f in A))} d'examen)", flush=True)
                acc[nom] = []
    print('FINI_EXTRACTION', f'{time.time() - t0:.0f} s', flush=True)

def lambda_dispatch(x):
    return bloc_m7(x[1]) if x[0] == 'm7' else cube_paris4(x[1])
if __name__ == '__main__':
    main()
