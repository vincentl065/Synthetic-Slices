"""Generate synthetic Herculaneum CT slices with exact per-sheet labels, as TRAINING DATA for other tools
(sheet / surface detectors, slice tracers, instance segmentation...).

Each tile = layout first (procedural, exact by construction) + texture second (conditional diffusion model trained on public scroll CT).
Outputs, in --out:
  shard_XXXX.npz  (one per batch, for training)  image uint8 (N, H, W) · lines uint8 (N, H, W) 0/1 (1-px median line of every sheet)
                  · sheet_id uint16 (N, H, W) (0 = no line, k = sheet k) · dist uint8 (N, H, W) (distance to nearest line, px, clipped 255)
                  · style int (N,) (texture style index, see synthetic_slices.STYLES) · seed int
  png/NNNNNN.png and png/NNNNNN_lines.png   (optional --png: the image, and the image with one colour per sheet; no text)

Examples:
  python generate.py --n 1000 --size 256 --out tiles256            # training tiles (fastest per pixel: one 256 px window)
  python generate.py --n 20 --size 512 --out canvases --png         # seamless 512 px canvases, also as PNG
  python generate.py --n 100 --style PHerc0813 --out t0813          # imitate one scroll's texture
Weights are downloaded from Hugging Face on first use (or pass --weights path/to/file.safetensors)."""
import argparse, os, time, numpy as np, torch
from scipy import ndimage as ndi
import synthetic_slices as S

HF_REPO = os.environ.get("SYNTH_SLICES_HF", "vincentl065/Synthetic-Slices")
PAL = [(240, 228, 66), (86, 180, 233), (204, 121, 167), (0, 158, 115), (230, 159, 0), (230, 120, 180)]      # RGB, cycled per sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16, help="number of tiles"); ap.add_argument("--size", type=int, default=256, help="tile size in px (multiple of 128, >= 256)")
    ap.add_argument("--out", default="out"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--style", default="random", help="random, or one of " + ", ".join(S.STYLES[:-1]))
    ap.add_argument("--batch", type=int, default=0, help="tiles denoised together (default: 16 for 256 px, 4 for 512 px)")
    ap.add_argument("--windows", type=int, default=9, help="256 px windows per network call (lower if GPU memory is short)")
    ap.add_argument("--steps", type=int, default=32, help="sampling steps (32 = released setting)"); ap.add_argument("--margin", type=int, default=48)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"]); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weights", default=None); ap.add_argument("--png", action="store_true", help="also write PNG previews")
    a = ap.parse_args(); assert a.size % 128 == 0 and a.size >= 256, "--size must be a multiple of 128, >= 256"
    os.makedirs(a.out, exist_ok=True)
    if a.png: os.makedirs(f"{a.out}/png", exist_ok=True)
    w = a.weights
    if w is None:
        from huggingface_hub import hf_hub_download
        w = hf_hub_download(HF_REPO, "synthetic_slices_ema_6655.safetensors")
    dev = a.device; net = S.load(w, dev)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}.get(a.dtype, "auto")
    if dt == "auto": dt = S.pick_dtype(dev)
    B = a.batch or (16 if a.size == 256 else 4); rng = np.random.default_rng(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    styles = [i for i, s in enumerate(S.STYLES) if s != "Paris4"]
    t0 = time.time(); k = 0; shard = 0
    while k < a.n:
        b = min(B, a.n - k); ids, lines, dist, sty = [], [], [], []
        for _ in range(b):
            m = S.sheet_layout(S.random_settings(rng), rng, (a.size, a.size), a.margin); ln = m > 0
            ids.append(m); lines.append(ln.astype(np.uint8)); dist.append(np.clip(np.round(ndi.distance_transform_edt(~ln)), 0, 255).astype(np.uint8))
            sty.append(rng.choice(styles) if a.style == "random" else S.STYLES.index(a.style))
        L = torch.tensor(np.stack(lines), device=dev); Dd = torch.tensor(np.stack(dist), device=dev).float()
        cond = S.condition(L, Dd); st = torch.tensor(sty, device=dev).long()
        img = S.to_uint8(S.sample(net, cond, st, a.size, a.size, gen, dev, dt, a.windows, a.steps))[:, 0].cpu().numpy()
        np.savez_compressed(f"{a.out}/shard_{shard:04d}.npz", image=img, lines=np.stack(lines), sheet_id=np.stack(ids), dist=np.stack(dist), style=np.array(sty), seed=a.seed)
        if a.png:
            import cv2
            for j in range(b):
                n = k + j; cv2.imwrite(f"{a.out}/png/{n:06d}.png", img[j]); c = np.repeat(img[j][..., None], 3, -1)
                for f in range(1, int(ids[j].max()) + 1):
                    mm = ids[j] == f
                    if mm.any(): c[mm] = PAL[f % len(PAL)]
                cv2.imwrite(f"{a.out}/png/{n:06d}_lines.png", c[..., ::-1])
        k += b; shard += 1; el = time.time() - t0
        print(f"{k}/{a.n} tiles · {el / k:.1f} s per tile · {el:.0f} s", flush=True)


if __name__ == "__main__":
    main()
