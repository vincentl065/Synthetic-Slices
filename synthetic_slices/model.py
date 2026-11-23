"""Conditional diffusion U-Net (step 2 of Synthetic-Slices): "sheet layout -> CT slice".

EDM formulation (Karras et al., 2022, "Elucidating the Design Space of Diffusion-Based Generative Models"):
the network gets a noisy slice x + n (noise level sigma) plus the condition and predicts the clean slice, with EDM preconditioning
(sigma_data = 0.5). Condition = 3 channels per pixel: [map present = 1, sheet lines (1 px) in {-1, +1}, distance to the nearest line
clipped at 32 px, scaled to [-1, 1]], plus embeddings of the noise level, the texture "style" (which scroll it imitates) and the
slicing axis. ~35 M parameters, 5 levels (64-128-192-256-256 channels), 2 residual blocks per level, self-attention at 32x32 and 16x16.
Attribute names are kept identical to the training code so the released weights load as they are."""
import math, torch, torch.nn as nn, torch.nn.functional as F

# texture styles = public ~9 um scans of 9 Herculaneum scrolls (neutral names) + Paris 4 with exact instance labels; style index NS = "unknown"
STYLES = ["style_0", "style_1", "style_2", "style_3", "style_4", "style_5", "style_6", "style_7", "style_8", "Paris4"]   # neutral names (same order and indices as v0.1)
NS = len(STYLES)
DMAX = 32.0
SIGMA_DATA = 0.5


def condition(lines, dist):
    """lines (B, H, W) 0/1, dist (B, H, W) in px -> (B, 3, H, W) in [-1, 1]"""
    lines = lines.float(); d = dist.float().clamp(max=DMAX)
    return torch.stack([torch.ones_like(lines), lines * 2 - 1, d / (DMAX / 2) - 1], 1)


class Bloc(nn.Module):
    """residual block with scale/shift from the embedding, optional multi-head self-attention"""
    def __init__(s, ci, co, de, attn=False):
        super().__init__()
        s.n1 = nn.GroupNorm(32, ci); s.c1 = nn.Conv2d(ci, co, 3, padding=1); s.e = nn.Linear(de, 2 * co)
        s.n2 = nn.GroupNorm(32, co); s.c2 = nn.Conv2d(co, co, 3, padding=1); s.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        s.attn = attn
        if attn: s.na = nn.GroupNorm(32, co); s.qkv = nn.Conv2d(co, 3 * co, 1); s.pr = nn.Conv2d(co, co, 1); s.nh = max(1, co // 64)

    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x)))
        a, b = s.e(e)[:, :, None, None].chunk(2, 1)
        h = s.c2(F.silu(s.n2(h) * (1 + a) + b)); x = s.sk(x) + h
        if s.attn:
            B, C, H, W = x.shape; q, k, v = s.qkv(s.na(x)).reshape(B, 3, s.nh, C // s.nh, H * W).transpose(-1, -2).unbind(1)
            o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, C, H, W); x = x + s.pr(o)
        return x


class UNet(nn.Module):
    def __init__(s, cin=1 + 3, ch=(64, 128, 192, 256, 256), nb=2, attn_res=(32, 16), de=512, res=256):
        super().__init__()
        s.de = de; s.f = nn.Sequential(nn.Linear(256, de), nn.SiLU(), nn.Linear(de, de))
        s.sty = nn.Embedding(NS + 1, de); s.axe = nn.Embedding(3, de)
        s.cin = nn.Conv2d(cin, ch[0], 3, padding=1)
        s.down, s.up = nn.ModuleList(), nn.ModuleList(); hs = [ch[0]]; c = ch[0]; r = res
        for i, co in enumerate(ch):
            for _ in range(nb): s.down.append(Bloc(c, co, de, r in attn_res)); c = co; hs.append(c)
            if i < len(ch) - 1: s.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); hs.append(c); r //= 2
        s.mid = nn.ModuleList([Bloc(c, c, de, True), Bloc(c, c, de, False)])
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(nb + 1): s.up.append(Bloc(c + hs.pop(), co, de, r in attn_res)); c = co
            if i > 0: s.up.append(nn.Upsample(scale_factor=2, mode="nearest")); s.up.append(nn.Conv2d(c, c, 3, padding=1)); r *= 2
        s.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def emb(s, cn, sty, axe):
        fr = torch.exp(-math.log(10000) * torch.arange(128, device=cn.device) / 128); a = cn[:, None] * 1000 * fr[None] / 4
        return s.f(torch.cat([a.sin(), a.cos()], 1)) + s.sty(sty) + s.axe(axe)

    def forward(s, x, cn, cond, sty, axe):
        e = s.emb(cn, sty, axe); h = s.cin(torch.cat([x, cond], 1)); hs = [h]
        for m in s.down:
            h = m(h, e) if isinstance(m, Bloc) else m(h); hs.append(h)
        for m in s.mid: h = m(h, e)
        for m in s.up:
            h = m(torch.cat([h, hs.pop()], 1), e) if isinstance(m, Bloc) else m(h)
        return s.out(h)


def denoise(net, x, sig, cond, sty, axe, dtype=None):
    """EDM-preconditioned denoiser D(x; sigma), sig (B,). dtype: autocast type on GPU (bf16 / fp16), None = full precision."""
    s = sig.view(-1, 1, 1, 1); cs = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
    co = s * SIGMA_DATA / (s ** 2 + SIGMA_DATA ** 2).sqrt(); ci = 1 / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    with torch.autocast(x.device.type, dtype or torch.float32, enabled=dtype is not None and x.device.type == "cuda"):
        Fx = net(x * ci, sig.log() / 4, cond, sty, axe).float()
    return cs * x + co * Fx


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def load(path, device="cuda"):
    """load released weights (.safetensors or .pt state dict)"""
    net = UNet().to(device).eval()
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(path), device=str(device) if device != "cuda" else "cuda")
    else:
        sd = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(sd); return net
