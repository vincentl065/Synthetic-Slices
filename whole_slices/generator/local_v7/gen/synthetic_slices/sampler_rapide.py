"""24/11 · PEINTRE RAPIDE (Vincent : « pousser sans perte la génération à sa vitesse maximale », des milliers de rouleaux 9 µm taille réelle).
Même modèle, mêmes poids, même principe (EDM + MultiDiffusion, fenêtres 256 px fondues par une fenêtre de Hann) que sampler.py, avec des leviers séparés :
  SANS PERTE (mêmes calculs, moins de gaspillage) :
    - fenêtres entièrement hors du rouleau non calculées (canal 0 de la condition = masque) ;
    - condition + sans-condition du guidage dans le même lot ; lots plus gros ; mémoire « channels_last » ; canevas en float32 sur la carte ;
  À VALIDER À L'ŒIL (changent le calcul) :
    - stride : recouvrement des fenêtres (128 = 4 passages par pixel, 192 = 1,8) ;
    - solveur : 'heun' (2 appels par pas, référence) ou 'dpmpp2m' (1 appel par pas, ordre 2 à pas multiples, sans bruit réinjecté) ;
    - steps : nombre de pas.
    sample(net, cond, style, H, W, gen, device, dtype, batch_windows, steps, guidance=..., stride=128, solver='heun', skip_empty=True)"""
import os, torch
from .model import denoise

T_WIN = 256


@torch.no_grad()
def sample(net, cond, style, H, W, gen, device="cuda", dtype=None, batch_windows=48, steps=12, churn=30.0, smin=0.05, smax=50.0, snoise=1.003,
           sigma_max=80.0, sigma_min=0.002, rho=7.0, guidance=1.0, stride=128, solver="heun", skip_empty=True, stats=None):
    assert cond.shape[0] == 1, "une coupe à la fois"
    ys = list(range(0, H - T_WIN + 1, stride)); xs = list(range(0, W - T_WIN + 1, stride))
    if ys[-1] != H - T_WIN: ys.append(H - T_WIN)
    if xs[-1] != W - T_WIN: xs.append(W - T_WIN)
    pos = [(y, x) for y in ys for x in xs]; n_tot = len(pos)
    if skip_empty: pos = [(y, x) for (y, x) in pos if bool((cond[0, 0, y:y + T_WIN, x:x + T_WIN] > 0).any())]
    N = len(pos); py = torch.tensor([p[0] for p in pos], device=device); px = torch.tensor([p[1] for p in pos], device=device)
    h1 = torch.hann_window(T_WIN + 2, periodic=False, device=device, dtype=torch.float64)[1:-1]; hann = torch.outer(h1, h1) + 1e-3
    wsum = torch.zeros(H, W, device=device, dtype=torch.float64)
    for (y, x) in pos: wsum[y:y + T_WIN, x:x + T_WIN] += hann
    couvert = wsum > 0; wsum = wsum.clamp(min=1e-6)
    t = torch.arange(steps, device=device, dtype=torch.float64)
    sig = (sigma_max ** (1 / rho) + t / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=device, dtype=torch.float64)])
    x = torch.randn(1, 1, H, W, device=device, generator=gen, dtype=torch.float64) * sig[0]
    sty = style.to(device); cfg = torch.is_tensor(guidance) or guidance != 1.0; appels = 0
    CL = os.environ.get('CL', '0') == '1'; net_cl = net.to(memory_format=torch.channels_last) if CL else net; mf = torch.channels_last if CL else torch.contiguous_format

    SPOT = os.environ.get('SPOT', '0') == '1'; GLO, GHI = float(os.environ.get('GUI_LO', 0.0)), float(os.environ.get('GUI_HI', 1e9)); plat = torch.ones(T_WIN, T_WIN, device=device, dtype=torch.float64)
    rs = torch.Generator().manual_seed(1234)

    def D(x, s):
        """SPOT=1 (SpotDiffusion, 2024) : fenêtres SANS recouvrement, décalées au hasard à chaque appel (la couture d'un pas est corrigée au suivant) = 1 passage par pixel
        au lieu de 4. GUI_LO / GUI_HI (Kynkäänniemi et al., 2024) : guidage seulement pour sigma dans [GUI_LO, GUI_HI] ; ailleurs un seul appel (condition seule)."""
        nonlocal appels
        if SPOT:
            oy, ox = int(torch.randint(0, T_WIN, (1,), generator=rs)), int(torch.randint(0, T_WIN, (1,), generator=rs))
            yl = sorted({min(max(v, 0), H - T_WIN) for v in range(-oy, H, T_WIN)}); xl = sorted({min(max(v, 0), W - T_WIN) for v in range(-ox, W, T_WIN)})
            P_all = [(a, b) for a in yl for b in xl if (not skip_empty) or bool((cond[0, 0, a:a + T_WIN, b:b + T_WIN] > 0).any())]; poids = plat
            ws = torch.zeros(H, W, device=device, dtype=torch.float64)
            for (a, b) in P_all: ws[a:a + T_WIN, b:b + T_WIN] += poids
            cv_ = ws > 0; ws = ws.clamp(min=1e-6)
        else:
            P_all, poids, ws, cv_ = pos, hann, wsum, couvert
        guide = cfg and GLO <= float(s) <= GHI
        acc = torch.zeros(H, W, device=device, dtype=torch.float64)
        for k in range(0, len(P_all), batch_windows):
            P = P_all[k:k + batch_windows]; n = len(P)
            xw = torch.stack([x[0, :, yy:yy + T_WIN, xx:xx + T_WIN] for (yy, xx) in P]).float(); cw = torch.stack([cond[0, :, yy:yy + T_WIN, xx:xx + T_WIN] for (yy, xx) in P])
            sv = torch.full((n,), float(s), device=device); sw = sty.expand(n); aw = torch.zeros(n, dtype=torch.long, device=device)
            if guide:
                dd = denoise(net_cl, torch.cat([xw, xw]).contiguous(memory_format=mf), torch.cat([sv, sv]), torch.cat([cw, torch.zeros_like(cw)]).contiguous(memory_format=mf), torch.cat([sw, sw]), torch.cat([aw, aw]), dtype)
                dc, du = dd[:n], dd[n:]
                g = torch.stack([guidance[0, :, yy:yy + T_WIN, xx:xx + T_WIN] for (yy, xx) in P]).float() if torch.is_tensor(guidance) else guidance
                d = du.double() + g * (dc.double() - du.double()); appels += 2 * n
            else:
                d = denoise(net_cl, xw.contiguous(memory_format=mf), sv, cw.contiguous(memory_format=mf), sw, aw, dtype).double(); appels += n
            d = d[:, 0] * poids
            for j, (yy, xx) in enumerate(P): acc[yy:yy + T_WIN, xx:xx + T_WIN] += d[j]
        return torch.where(cv_, acc / ws, torch.zeros_like(acc))[None, None]

    if solver == "heun":
        for i in range(steps):
            sc, sn = sig[i], sig[i + 1]; gm = min(churn / steps, 2 ** 0.5 - 1) if smin <= float(sc) <= smax else 0.0; sh = sc * (1 + gm)
            if gm > 0: x = x + (sh ** 2 - sc ** 2).sqrt() * snoise * torch.randn(x.shape, device=device, generator=gen, dtype=torch.float64)
            d = (x - D(x, sh)) / sh; xn = x + (sn - sh) * d
            if sn > 0: d2 = (xn - D(xn, sn)) / sn; xn = x + (sn - sh) * (0.5 * d + 0.5 * d2)
            x = xn
    elif solver == "dpmpp2m":
        prev = None; hp = None
        for i in range(steps):
            sc, sn = sig[i], sig[i + 1]; den = D(x, sc)
            if float(sn) == 0: x = den; break
            h = torch.log(sc / sn)
            if prev is None: dt_ = den
            else: r = hp / h; dt_ = (1 + 1 / (2 * r)) * den - (1 / (2 * r)) * prev
            x = (sn / sc) * x - torch.expm1(-h) * dt_; prev = den; hp = h
    else:
        raise ValueError(solver)
    if stats is not None: stats.update(fenetres=N, fenetres_total=n_tot, appels_reseau=appels)
    return x.float().clamp(-1, 1)
