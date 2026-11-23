"""Shared training code: scroll list (public ~9 um scans), data loading, condition, U-Net, EDM denoiser and sampler.
Outputs/inputs live under $GEN_RACINE (default /workspace/generateur): donnees/ (pairs), points/ (checkpoints), planches/ (previews)."""
import os, json, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
RACINE = os.environ.get('GEN_RACINE', '/workspace/generateur')
DONNEES = f'{RACINE}/donnees'
ROULEAUX = {'style_0': '20250821151803-9.362um-1.2m-113keV', 'style_1': '20250821151648-9.362um-1.2m-113keV', 'style_2': '20250820131727-9.362um-1.2m-113keV', 'style_3': '20250521115057-8.640um-1.2m-116keV', 'style_4': '20250821151737-9.362um-1.2m-113keV', 'style_5': '20250821151723-9.362um-1.2m-113keV', 'style_6': '20250821151701-9.362um-1.2m-113keV', 'style_7': '20250821151635-9.362um-1.2m-113keV', 'style_8': '20250521133212-8.640um-1.2m-116keV'}   # neutral names, same order as v0.1; values = public scan volume ids
STYLES = list(ROULEAUX) + ['Paris4']
NS = len(STYLES)
DMAX = 32.0

def journal(txt):
    with open(f'{RACINE}/journal.md', 'a', encoding='utf-8') as f:
        f.write(txt.rstrip() + '\n')

def charger_jeu(test=None):
    import glob
    P = {k: [] for k in ('ct', 'lig', 'dist', 'style', 'axe', 'test', 'bloc')}
    for f in sorted(glob.glob(f'{DONNEES}/tuiles_*.npz')):
        z = np.load(f)
        P['ct'].append(z['ct'])
        P['lig'].append(np.unpackbits(z['lig'], axis=2)[:, :, :256])
        P['dist'].append(z['dist'])
        for k in ('style', 'axe', 'test', 'bloc'):
            P[k].append(z[k])
    P = {k: np.concatenate(v) for k, v in P.items()}
    if test is not None:
        m = P['test'] == test
        P = {k: v[m] for k, v in P.items()}
    return P

def condition(lig, dist):
    lig = lig.float()
    d = dist.float().clamp(max=DMAX)
    return torch.stack([torch.ones_like(lig), lig * 2 - 1, d / (DMAX / 2) - 1], 1)

class Bloc(nn.Module):

    def __init__(s, ci, co, de, attn=False):
        super().__init__()
        s.n1 = nn.GroupNorm(32, ci)
        s.c1 = nn.Conv2d(ci, co, 3, padding=1)
        s.e = nn.Linear(de, 2 * co)
        s.n2 = nn.GroupNorm(32, co)
        s.c2 = nn.Conv2d(co, co, 3, padding=1)
        s.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        nn.init.zeros_(s.c2.weight)
        nn.init.zeros_(s.c2.bias)
        s.attn = attn
        if attn:
            s.na = nn.GroupNorm(32, co)
            s.qkv = nn.Conv2d(co, 3 * co, 1)
            s.pr = nn.Conv2d(co, co, 1)
            s.nh = max(1, co // 64)
            nn.init.zeros_(s.pr.weight)
            nn.init.zeros_(s.pr.bias)

    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x)))
        a, b = s.e(e)[:, :, None, None].chunk(2, 1)
        h = s.c2(F.silu(s.n2(h) * (1 + a) + b))
        x = s.sk(x) + h
        if s.attn:
            B, C, H, W = x.shape
            q, k, v = s.qkv(s.na(x)).reshape(B, 3, s.nh, C // s.nh, H * W).transpose(-1, -2).unbind(1)
            o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, C, H, W)
            x = x + s.pr(o)
        return x

class UNet(nn.Module):

    def __init__(s, cin=1 + 3, ch=(64, 128, 192, 256, 256), nb=2, attn_res=(32, 16), de=512, res=256):
        super().__init__()
        s.de = de
        s.f = nn.Sequential(nn.Linear(256, de), nn.SiLU(), nn.Linear(de, de))
        s.sty = nn.Embedding(NS + 1, de)
        s.axe = nn.Embedding(3, de)
        s.cin = nn.Conv2d(cin, ch[0], 3, padding=1)
        s.down, s.up = (nn.ModuleList(), nn.ModuleList())
        hs = [ch[0]]
        c = ch[0]
        r = res
        for i, co in enumerate(ch):
            for _ in range(nb):
                s.down.append(Bloc(c, co, de, r in attn_res))
                s.down[-1].ck = r >= 128
                c = co
                hs.append(c)
            if i < len(ch) - 1:
                s.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                hs.append(c)
                r //= 2
        s.mid = nn.ModuleList([Bloc(c, c, de, True), Bloc(c, c, de, False)])
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(nb + 1):
                s.up.append(Bloc(c + hs.pop(), co, de, r in attn_res))
                s.up[-1].ck = r >= 128
                c = co
            if i > 0:
                s.up.append(nn.Upsample(scale_factor=2, mode='nearest'))
                s.up.append(nn.Conv2d(c, c, 3, padding=1))
                r *= 2
        s.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(s.out[-1].weight)
        nn.init.zeros_(s.out[-1].bias)

    def emb(s, cn, sty, axe):
        fr = torch.exp(-math.log(10000) * torch.arange(128, device=cn.device) / 128)
        a = cn[:, None] * 1000 * fr[None] / 4
        return s.f(torch.cat([a.sin(), a.cos()], 1)) + s.sty(sty) + s.axe(axe)

    def _b(s, m, h, e):
        if s.training and getattr(m, 'ck', False):
            return torch.utils.checkpoint.checkpoint(m, h, e, use_reentrant=False)
        return m(h, e)

    def forward(s, x, cn, cond, sty, axe):
        e = s.emb(cn, sty, axe)
        h = s.cin(torch.cat([x, cond], 1))
        hs = [h]
        for m in s.down:
            h = s._b(m, h, e) if isinstance(m, Bloc) else m(h)
            hs.append(h)
        for m in s.mid:
            h = m(h, e)
        for m in s.up:
            if isinstance(m, Bloc):
                h = s._b(m, torch.cat([h, hs.pop()], 1), e)
            else:
                h = m(h)
        return s.out(h)
SD = 0.5

def debruiter(net, x, sig, cond, sty, axe):
    s = sig.view(-1, 1, 1, 1)
    cs = SD ** 2 / (s ** 2 + SD ** 2)
    co = s * SD / (s ** 2 + SD ** 2).sqrt()
    ci = 1 / (s ** 2 + SD ** 2).sqrt()
    with torch.autocast('cuda', torch.bfloat16):
        Fx = net(x * ci, sig.log() / 4, cond, sty, axe).float()
    return cs * x + co * Fx

@torch.no_grad()
def echantillonner(net, cond, sty, axe, n_pas=32, g=None, guide=1.0, churn=0.0, smin=0.05, smax=50.0, snoise=1.003, smx=80.0, smn=0.002, rho=7.0):
    B = cond.shape[0]
    dev = cond.device
    t = torch.arange(n_pas, device=dev, dtype=torch.float64)
    sig = (smx ** (1 / rho) + t / (n_pas - 1) * (smn ** (1 / rho) - smx ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev, dtype=torch.float64)])
    x = torch.randn(B, 1, cond.shape[2], cond.shape[3], device=dev, generator=g, dtype=torch.float64) * sig[0]

    def D(x, s):
        sv = torch.full((B,), float(s), device=dev)
        d = debruiter(net, x.float(), sv, cond, sty, axe)
        if guide != 1.0:
            du = debruiter(net, x.float(), sv, torch.zeros_like(cond), sty, axe)
            d = du + guide * (d - du)
        return d.double()
    for i in range(n_pas):
        sc, sn = (sig[i], sig[i + 1])
        gm = min(churn / n_pas, 2 ** 0.5 - 1) if smin <= sc <= smax else 0.0
        sh = sc * (1 + gm)
        if gm > 0:
            x = x + (sh ** 2 - sc ** 2).sqrt() * snoise * torch.randn(x.shape, device=dev, generator=g, dtype=torch.float64)
        d = (x - D(x, sh)) / sh
        xn = x + (sn - sh) * d
        if sn > 0:
            d2 = (xn - D(xn, sn)) / sn
            xn = x + (sn - sh) * (0.5 * d + 0.5 * d2)
        x = xn
    return x.float().clamp(-1, 1)

def vers_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
