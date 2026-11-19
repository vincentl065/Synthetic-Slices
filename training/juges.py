"""The 3 judges of the generator: (1) REALISM: 3 small real-vs-generated classifiers on 96-px crops, AUC on held-out tiles
(50 % = indistinguishable), also on blurred (shape) and high-pass (grain) versions; (2) LAYOUT FIDELITY: a sheet detector trained on
REAL tiles only, F1 at 2 px between what it finds and the layout given to the generator; (3) NO COPY: cosine similarity to the
nearest training tile (8 symmetries) compared with the same measure for real held-out tiles."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

def auc(pv, pf):
    r = np.argsort(np.argsort(np.r_[pv, pf])) + 1
    return float((r[len(pv):].sum() - len(pf) * (len(pf) + 1) / 2) / (len(pv) * len(pf)))

def _mode(t, mode):
    if mode == 'brut':
        return t
    a = t.astype(np.float32)
    f = ndi.gaussian_filter(a, (0, 2, 2))
    return np.clip(f, 0, 255).astype(np.uint8) if mode == 'forme' else np.clip(a - f + 128, 0, 255).astype(np.uint8)

def juge_realisme(vraies, fausses, mode='brut', dev='cuda', graine=0):
    rng = np.random.default_rng(graine)
    P = 96
    vraies = _mode(vraies, mode)
    fausses = _mode(fausses, mode)

    def coupes(T, n_par):
        X, idx = ([], [])
        for i, t in enumerate(T):
            for _ in range(n_par):
                y, x = (rng.integers(0, t.shape[0] - P + 1), rng.integers(0, t.shape[1] - P + 1))
                a = t[y:y + P, x:x + P]
                if rng.random() < 0.5:
                    a = a[:, ::-1]
                X.append(np.rot90(a, rng.integers(4)).copy())
                idx.append(i)
        return (np.array(X, np.float32)[:, None] / 255, np.array(idx))
    nv, nf = (len(vraies), len(fausses))
    tv, tf = (rng.permutation(nv), rng.permutation(nf))
    va, vt = (tv[:int(0.7 * nv)], tv[int(0.7 * nv):])
    fa, ft = (tf[:int(0.7 * nf)], tf[int(0.7 * nf):])
    npv = 4
    npf = max(1, int(round(npv * len(va) / max(1, len(fa)))))
    Xv, _ = coupes(vraies[va], npv)
    Xf, _ = coupes(fausses[fa], npf)
    X = torch.tensor(np.concatenate([Xv, Xf]), device=dev)
    Y = torch.tensor(np.r_[np.zeros(len(Xv)), np.ones(len(Xf))], dtype=torch.float32, device=dev)
    Tv, _ = coupes(vraies[vt], 4)
    Tf, jf = coupes(fausses[ft], 4)
    Tv, Tf = (torch.tensor(Tv, device=dev), torch.tensor(Tf, device=dev))
    A, PF = ([], [])
    for gr in range(3):
        torch.manual_seed(gr)
        net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 0.001)
        for ep in range(8):
            perm = torch.randperm(len(X), device=dev)
            for b in range(0, len(X), 64):
                j = perm[b:b + 64]
                l = F.binary_cross_entropy_with_logits(net(X[j]).squeeze(1), Y[j])
                opt.zero_grad()
                l.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Tv)).squeeze(1).cpu().numpy()
            pf = torch.sigmoid(net(Tf)).squeeze(1).cpu().numpy()
        A.append(auc(pv, pf))
        PF.append(pf)
    pf = np.mean(PF, 0)
    sc = np.array([pf[jf == j].mean() for j in range(len(ft))])
    return (float(np.mean(A)), [float(a) for a in A], ft, sc)

def bloc(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(), nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())

class Det(nn.Module):

    def __init__(s, c=32):
        super().__init__()
        ch = [c, 2 * c, 4 * c, 8 * c]
        s.e = nn.ModuleList([bloc(1, ch[0]), bloc(ch[0], ch[1]), bloc(ch[1], ch[2])])
        s.m = bloc(ch[2], ch[3])
        s.u = nn.ModuleList([nn.ConvTranspose2d(ch[3], ch[2], 2, 2), nn.ConvTranspose2d(ch[2], ch[1], 2, 2), nn.ConvTranspose2d(ch[1], ch[0], 2, 2)])
        s.d = nn.ModuleList([bloc(2 * ch[2], ch[2]), bloc(2 * ch[1], ch[1]), bloc(2 * ch[0], ch[0])])
        s.o = nn.Conv2d(ch[0], 1, 1)

    def forward(s, x):
        k = []
        for e in s.e:
            x = e(x)
            k.append(x)
            x = F.avg_pool2d(x, 2)
        x = s.m(x)
        for u, d, kk in zip(s.u, s.d, reversed(k)):
            x = d(torch.cat([u(x), kk], 1))
        return s.o(x)

def entrainer_detecteur(ct, lig, pas=4000, c=32, dev='cuda', log=print):
    torch.manual_seed(11)
    det = Det(c).to(dev).to(memory_format=torch.channels_last)
    o = torch.optim.AdamW(det.parameters(), 0.001, weight_decay=0.0001)
    sch = torch.optim.lr_scheduler.OneCycleLR(o, 0.001, total_steps=pas, pct_start=0.05)
    for p in range(pas):
        i = torch.randint(0, len(ct), (16,), device=ct.device)
        x = ct[i].to(dev).float()[:, None] / 127.5 - 1
        y = lig[i].to(dev).float()[:, None]
        y = F.max_pool2d(y, 3, 1, 1)
        k = int(torch.randint(0, 4, (1,)))
        x, y = (torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3)))
        if torch.rand(1) < 0.5:
            x, y = (x.flip(3), y.flip(3))
        with torch.autocast('cuda', torch.bfloat16):
            out = det(x)
        l = F.binary_cross_entropy_with_logits(out.float(), y, pos_weight=torch.tensor(3.0, device=dev))
        o.zero_grad(set_to_none=True)
        l.backward()
        o.step()
        sch.step()
        if p % 1000 == 0:
            log(f'détecteur pas {p} perte {l.item():.3f}')
    return det.eval()

def f1(pred, ref, tol=2):
    p, r = (pred > 0, ref > 0)
    if not r.any() and (not p.any()):
        return 1.0
    rd, pd = (ndi.binary_dilation(r, iterations=tol), ndi.binary_dilation(p, iterations=tol))
    prec = (p & rd).sum() / max(1, p.sum())
    rap = (r & pd).sum() / max(1, r.sum())
    return float(2 * prec * rap / max(1e-09, prec + rap))

@torch.no_grad()
def fidelite(det, imgs, ligs, dev='cuda', bande=False):
    s = []
    for b in range(0, len(imgs), 64):
        x = torch.tensor(imgs[b:b + 64], device=dev).float()[:, None] / 127.5 - 1
        with torch.autocast('cuda', torch.bfloat16):
            p = (torch.sigmoid(det(x).float())[:, 0] > 0.5).cpu().numpy()
        if bande:
            s += [f1(p[i], ndi.binary_dilation(ligs[b + i] > 0, structure=np.ones((3, 3), bool))) for i in range(len(p))]
        else:
            s += [f1(skeletonize(p[i]), ligs[b + i]) for i in range(len(p))]
    return (float(np.mean(s)), np.array(s))

def signatures(u8, dev='cuda'):
    out = []
    for b in range(0, len(u8), 1024):
        x = torch.as_tensor(u8[b:b + 1024]).to(dev).float()[:, None]
        x = F.avg_pool2d(x, 2)[:, 0].flatten(1)
        x = x - x.mean(1, keepdim=True)
        out.append((x / (x.norm(dim=1, keepdim=True) + 1e-06)).half())
    return torch.cat(out)

@torch.no_grad()
def plus_proche(u8, sig_ent, dev='cuda'):
    best = torch.full((len(u8),), -1.0, device=dev)
    t = torch.as_tensor(u8, device=dev)
    for k in range(4):
        for fl in (False, True):
            v = torch.rot90(t, k, (1, 2))
            v = v.flip(2) if fl else v
            s = signatures(v, dev)
            for b in range(0, len(sig_ent), 4096):
                best = torch.maximum(best, (s @ sig_ent[b:b + 4096].T).float().max(1).values)
    return best.cpu().numpy()

@torch.no_grad()
def details(det, imgs, ligs, dev='cuda', tol=3):
    import cartes as C
    out = {k: [] for k in ('trait_sans_feuille', 'feuille_sans_trait', 'trait_sur_vide', 'papyrus_sans_trait', 'cont_det', 'formes_det', 'boucles_det')}
    for b in range(0, len(imgs), 64):
        x = torch.tensor(imgs[b:b + 64], device=dev).float()[:, None] / 127.5 - 1
        with torch.autocast('cuda', torch.bfloat16):
            p = (torch.sigmoid(det(x).float())[:, 0] > 0.5).cpu().numpy()
        for i in range(len(p)):
            l = ligs[b + i] > 0
            sp = skeletonize(p[i])
            dl = ndi.distance_transform_edt(~l) if l.any() else np.full(l.shape, 99.0)
            dp = ndi.distance_transform_edt(~sp) if sp.any() else np.full(l.shape, 99.0)
            out['trait_sans_feuille'].append(float((dp[l] > tol).mean()) if l.any() else 0.0)
            out['feuille_sans_trait'].append(float((dl[sp] > tol).mean()) if sp.any() else 0.0)
            pap = C.papyrus(imgs[b + i])
            dpa = ndi.distance_transform_edt(~pap) if pap.any() else np.full(l.shape, 99.0)
            out['trait_sur_vide'].append(float((dpa[l] > 2).mean()) if l.any() else 0.0)
            out['papyrus_sans_trait'].append(float((dl[pap] > 10).mean()) if pap.any() else 0.0)
            m, _ = C.mesures(sp)
            out['cont_det'].append(m['cont'])
            out['formes_det'].append(m.get('formes', 0.0))
            out['boucles_det'].append(m['boucles'])
    return {k: round(float(np.mean(v)), 3) for k, v in out.items()}

def continuite_cartes(ligs):
    import cartes as C
    M = [C.mesures(l)[0] for l in ligs]
    return dict(cont=round(float(np.mean([m['cont'] for m in M])), 3), rejet=round(float(np.mean([bool(C.rejet(m)) for m in M])), 3), boucles=round(float(np.mean([m['boucles'] for m in M])), 2))
