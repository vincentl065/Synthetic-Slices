"""24/11 · TRACÉ ABÎMÉ pour le peintre (essai_trace_abime.py : coupures + petits bouts épars -> le générateur peint la masse dense et chaotique des zones écrasées ;
tracé parfait -> feuilles trop sages, reconnues par Vincent au test à l'aveugle). La VÉRITÉ (champ exact) ne change pas ; les plages écrasées sont rendues à part
pour être comptées « illisibles ».
    L2, plages = abimer(L, dedans, graine, part_plages=0.4, ...)"""
import numpy as np, cv2
from scipy import ndimage as ndi


def _champ(r, shape, sig, f=8):
    h, w = shape; g = cv2.GaussianBlur(r.normal(0, 1, (h // f + 2, w // f + 2)).astype(np.float32), (0, 0), max(0.6, sig / f))
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); return g / (g.std() + 1e-6)


def abimer(L, ded, graine, part_plages=0.4, taille_plages=260, tremble=1.0, coup_base=0.04, coup_plage=0.22, bouts_base=0.0, bouts_plage=10.0):
    """L : tracé (bool) ; ded : intérieur du rouleau. Rend (tracé abîmé, carte des plages écrasées 0–1)."""
    from skimage.morphology import skeletonize
    r = np.random.default_rng(graine); H, W = L.shape
    pl = np.zeros((H, W), np.float32)
    if part_plages > 0:
        g = _champ(r, (H, W), taille_plages, 16); s = np.quantile(g[ded][::7], 1 - part_plages); pl = cv2.GaussianBlur((np.clip((g - s) / 0.25 + 0.5, 0, 1) * ded).astype(np.float32), (0, 0), 16)
    A = L.copy()
    if tremble > 0:                                                              # tremblé (longueurs d'onde ~12–40 px), un peu plus fort dans les plages
        k = tremble * 1.8 * pl                                                  # 24/11 10 h 40 : tremblé SEULEMENT dans les plages (ailleurs il décalait l'image de la vérité : suivi 0,25 -> 0,11–0,16)
        dx = (0.6 * _champ(r, (H, W), 5, 2) * 12.5 + 0.4 * _champ(r, (H, W), 14, 4) * 35) * 0.12 * k; dy = (0.6 * _champ(r, (H, W), 5, 2) * 12.5 + 0.4 * _champ(r, (H, W), 14, 4) * 35) * 0.12 * k
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); A = skeletonize(cv2.remap(cv2.dilate(L.astype(np.uint8), np.ones((2, 2), np.uint8)), xx + dx, yy + dy, cv2.INTER_NEAREST) > 0); del dx, dy, yy, xx
    f = _champ(r, (H, W), 7, 2); u = (f - f.min()) / (f.max() - f.min()); rang = ndi.uniform_filter(u, 1)                # coupures : part locale = coup_base … coup_plage
    q = np.quantile(f[::5, ::5], 1 - np.linspace(0.01, 0.5, 50)); part = coup_base + (coup_plage - coup_base) * pl; seuil = np.interp(part, np.linspace(0.01, 0.5, 50), q); A &= ~(f > seuil); del f, u, rang, seuil
    d = ndi.distance_transform_edt(~L); gy, gx = np.gradient(cv2.GaussianBlur(np.minimum(d, 12).astype(np.float32), (0, 0), 2)); ok = (d > 3.5) & (d < 9) & ded     # bouts épars
    dens = (bouts_base + (bouts_plage - bouts_base) * pl) / 1e4; ys, xs = np.nonzero(ok & (r.random((H, W)) < dens / max(float(ok.mean()), 1e-6)))        # nombre de bouts par 10 000 px de surface = bouts_base … bouts_plage
    B = np.zeros((H, W), np.uint8)
    for y, x in zip(ys, xs):
        t = np.arctan2(gx[y, x], -gy[y, x]) + r.normal(0, 0.35); l = r.uniform(6, 26); cv2.line(B, (int(x - l * np.cos(t)), int(y - l * np.sin(t))), (int(x + l * np.cos(t)), int(y + l * np.sin(t))), 1, 1)
    return (A | (B > 0)) & ded, pl
