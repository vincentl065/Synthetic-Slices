"""Sheet-line detector trained on synthetic tiles only (v0.2 benchmark arm "S5").

A small 2D U-Net (3 levels, 32-64-128 channels, ~1.3 M parameters) that maps a CT slice to the probability of "median line of a sheet".
It was trained on 2,000 synthetic 256 px tiles from the v0.2 recipe (synthetic_slices.v3.RECIPE_S5, painted by the generator retrained
without any Paris 4 data), 3,000 steps, batch 16, AdamW 1e-3 OneCycle; the best step was chosen on 56 held-out real tiles.
Scale: ~9.4 um per pixel. Each 256 px window is normalised by its own 1st / 99th percentiles, so the input grey range does not matter.

    net = load_detector(path, device)          # .safetensors or .pt state dict
    prob = read_slice(net, image, device)      # (H, W) float32 in [0, 1], overlapping 256 px windows blended with a Hann weight
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def _block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())


class Det(nn.Module):
    """attribute names kept identical to the training code so the released weights load as they are"""
    def __init__(s, c=32, niv=3, nout=1):
        super().__init__(); ch = [c * 2 ** k for k in range(niv)]
        s.e = nn.ModuleList([_block(1, ch[0])] + [_block(ch[k - 1], ch[k]) for k in range(1, niv)]); s.m = _block(ch[-1], ch[-1])
        s.u = nn.ModuleList([nn.ConvTranspose2d(ch[-1], ch[-1], 2, 2)] + [nn.ConvTranspose2d(ch[k], ch[k - 1], 2, 2) for k in range(niv - 1, 0, -1)])
        s.d = nn.ModuleList([_block(2 * ch[k], ch[k]) for k in range(niv - 1, -1, -1)]); s.o = nn.Conv2d(ch[0], nout, 1)

    def forward(s, x):
        k = []
        for e in s.e: x = e(x); k.append(x); x = F.avg_pool2d(x, 2)
        x = s.m(x)
        for u, dd, kk in zip(s.u, s.d, reversed(k)): x = dd(torch.cat([u(x), kk], 1))
        return s.o(x)


def norm(t):
    """per-window normalisation: 1st / 99th percentiles -> [-1, 1] (clipped at +-1.5)"""
    q = torch.quantile(t.flatten(1)[:, ::7], torch.tensor([0.01, 0.99], device=t.device), dim=1)
    return ((t - q[0][:, None, None]) / (q[1] - q[0] + 1e-3)[:, None, None] * 2 - 1).clamp(-1.5, 1.5)


def load_detector(path, device="cpu"):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    net = Det(32, 3, sd["o.weight"].shape[0]); net.load_state_dict(sd); return net.to(device).eval()


@torch.no_grad()
def read_slice(net, X, device="cpu", T=256, batch=32):
    """X (H, W) any numeric type -> (H, W) float32 probability of the first output (sheet line)"""
    X = np.asarray(X, np.float32); H, W = X.shape
    if H < T or W < T: X = np.pad(X, ((0, max(0, T - H)), (0, max(0, T - W))), mode="reflect")
    Hh, Ww = X.shape; acc = np.zeros((Hh, Ww), np.float32); wsum = np.zeros((Hh, Ww), np.float32)
    w1 = np.hanning(T + 2)[1:-1]; hw = np.outer(w1, w1).astype(np.float32) + 1e-3
    ys = sorted(set(list(range(0, Hh - T + 1, T // 2)) + [Hh - T])); xs = sorted(set(list(range(0, Ww - T + 1, T // 2)) + [Ww - T]))
    win = [(a, b) for a in ys for b in xs]
    for k0 in range(0, len(win), batch):
        bl = win[k0:k0 + batch]; x = torch.from_numpy(np.stack([X[a:a + T, b:b + T] for a, b in bl])).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            pr = torch.sigmoid(net(norm(x)[:, None]).float())[:, 0].cpu().numpy()
        for (a, b), q in zip(bl, pr): acc[a:a + T, b:b + T] += q * hw; wsum[a:a + T, b:b + T] += hw
    return (acc / np.maximum(wsum, 1e-6))[:H, :W]
