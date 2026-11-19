"""Seamless canvases with the EDM sampler (Heun, 2nd order, stochastic "churn") + MultiDiffusion (Bar-Tal et al., 2023).

At every denoising step the H x W canvas is split into 256 x 256 windows overlapping by half (stride 128); each window is denoised with
its piece of the condition, and the results are blended with a Hann weight -> one image without seams. A 256 x 256 tile is one window.
Settings used for the released examples: 32 noise levels from sigma 80 down to 0.002 (Karras schedule, rho 7), churn 30 between
sigma 0.05 and 50, s_noise 1.003. Cost: 2 network calls per step (except the last) per window."""
import torch
from .model import denoise

T_WIN, STRIDE = 256, 128


def pick_dtype(device):
    """fp16 on AMD (RX 7800 XT, ROCm: 1.6x faster than bf16), bf16 on NVIDIA GPUs with native bf16 (Ampere and newer), fp16 otherwise
    (older NVIDIA cards emulate bf16). fp16 gives the same images as bf16 (RX 7800 XT, same seed: mean absolute difference 0.15 grey
    level out of 255, correlation 0.9999)."""
    if device != "cuda" or not torch.cuda.is_available(): return None
    if getattr(torch.version, "hip", None): return torch.float16
    try:
        if torch.cuda.get_device_capability()[0] >= 8: return torch.bfloat16
    except Exception: pass
    return torch.float16


@torch.no_grad()
def sample(net, cond, style, H, W, gen, device="cuda", dtype=None, batch_windows=9, steps=32, churn=30.0,
           smin=0.05, smax=50.0, snoise=1.003, sigma_max=80.0, sigma_min=0.002, rho=7.0):
    """cond (B, 3, H, W), style (B,) long -> images in [-1, 1], (B, 1, H, W). H, W: multiples of 128, >= 256."""
    B = cond.shape[0]; pos = [(y, x) for y in range(0, H - T_WIN + 1, STRIDE) for x in range(0, W - T_WIN + 1, STRIDE)]
    h1 = torch.hann_window(T_WIN + 2, periodic=False, device=device, dtype=torch.float64)[1:-1]; hann = torch.outer(h1, h1) + 1e-3
    t = torch.arange(steps, device=device, dtype=torch.float64)
    sig = (sigma_max ** (1 / rho) + t / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=device, dtype=torch.float64)])
    x = torch.randn(B, 1, H, W, device=device, generator=gen, dtype=torch.float64) * sig[0]
    cw = torch.stack([cond[b, :, yy:yy + T_WIN, xx:xx + T_WIN] for b in range(B) for (yy, xx) in pos])
    sw = style.repeat_interleave(len(pos)); aw = torch.zeros(len(sw), dtype=torch.long, device=device)

    def D(x, s):
        xw = torch.stack([x[b, :, yy:yy + T_WIN, xx:xx + T_WIN] for b in range(B) for (yy, xx) in pos]).float()
        sv = torch.full((len(xw),), float(s), device=device); K = batch_windows
        d = torch.cat([denoise(net, xw[k:k + K], sv[k:k + K], cw[k:k + K], sw[k:k + K], aw[k:k + K], dtype) for k in range(0, len(xw), K)]).double()
        acc = torch.zeros_like(x); w = torch.zeros_like(x); j = 0
        for b in range(B):
            for (yy, xx) in pos:
                acc[b, :, yy:yy + T_WIN, xx:xx + T_WIN] += d[j] * hann; w[b, :, yy:yy + T_WIN, xx:xx + T_WIN] += hann; j += 1
        return acc / w

    for i in range(steps):
        sc, sn = sig[i], sig[i + 1]; gm = min(churn / steps, 2 ** 0.5 - 1) if smin <= sc <= smax else 0.0; sh = sc * (1 + gm)
        if gm > 0: x = x + (sh ** 2 - sc ** 2).sqrt() * snoise * torch.randn(x.shape, device=device, generator=gen, dtype=torch.float64)
        d = (x - D(x, sh)) / sh; xn = x + (sn - sh) * d
        if sn > 0: d2 = (xn - D(xn, sn)) / sn; xn = x + (sn - sh) * (0.5 * d + 0.5 * d2)
        x = xn
    return x.float().clamp(-1, 1)
