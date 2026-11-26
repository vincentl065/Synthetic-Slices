"""Command line: `synthetic-slices generate ...` and `synthetic-slices detect ...` (also `python -m synthetic_slices ...`).

generate  synthetic CT tiles with exact per-sheet labels (training data), written as shard_XXXX.npz (+ optional PNG previews)
detect    sheet lines on a CT slice with the detector trained on synthetic tiles only, extracted with NMS 2 (one line per sheet, no branch)
"""
import argparse, os, sys, time
import numpy as np

PAL = [(240, 228, 66), (86, 180, 233), (204, 121, 167), (0, 158, 115), (230, 159, 0), (0, 114, 178)]   # Okabe-Ito (RGB), cycled per sheet


# ----------------------------------------------------------------------------------------------------------------------------- generate
def cmd_generate(a):
    import torch
    from scipy import ndimage as ndi
    import synthetic_slices as S
    from . import v3, hub
    assert a.size % 128 == 0 and a.size >= 256, "--size must be a multiple of 128, >= 256"
    if a.threads: torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.png: os.makedirs(os.path.join(a.out, "png"), exist_ok=True)
    w = hub.resolve(a.weights, hub.GENERATORS); dev = a.device; net = S.load(w, dev)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}.get(a.dtype, "auto")
    if dt == "auto": dt = S.pick_dtype(dev)
    B = a.batch or (16 if a.size == 256 else 4)
    if dev == "cpu" and not a.batch: B = 4
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device=dev).manual_seed(a.seed)
    names = [s for s in S.STYLES if s != "Paris4"]
    if a.style == "random": pick = lambda: S.STYLES.index(names[int(rng.integers(len(names)))])
    elif a.style == "unknown": pick = lambda: S.NS
    elif a.style.isdigit(): pick = lambda: int(a.style)
    else: pick = lambda: S.STYLES.index(a.style)
    print(f"generator {os.path.basename(w)} · recipe {a.recipe} · {a.size} px · {a.steps} steps · device {dev} · batch {B}", flush=True)
    t0 = time.time(); k = 0; shard = 0
    while k < a.n:
        b = min(B, a.n - k); lays = []
        for j in range(b):
            if a.recipe == "v0.1":
                m = S.sheet_layout(S.random_settings(rng), rng, (a.size, a.size), a.margin); ln = m > 0; d = ndi.distance_transform_edt(~ln)
                lays.append(dict(sheet_id=m.astype(np.uint16), lines=ln.astype(np.uint8), dist=np.clip(np.round(d), 0, 255).astype(np.uint8),
                                 dist_cond=(np.clip(np.round(d * 4), 0, 255) / 4).astype(np.float32), crack=np.zeros(ln.shape, np.uint8),
                                 air=np.zeros(ln.shape, bool), crushed=0, kind="plain", guidance=1.0))
            else:
                lays.append(v3.random_layout(a.seed, k + j, a.size, a.margin, v3.RECIPE_S5))
        sty = [pick() for _ in range(b)]; img = np.zeros((b, a.size, a.size), np.uint8)
        for g in sorted({x["guidance"] for x in lays}):
            idx = [j for j in range(b) if lays[j]["guidance"] == g]
            L = torch.tensor(np.stack([lays[j]["lines"] for j in idx]), device=dev); Dd = torch.tensor(np.stack([lays[j]["dist_cond"] for j in idx]), device=dev)
            out = S.sample(net, S.condition(L, Dd), torch.tensor([sty[j] for j in idx], device=dev).long(), a.size, a.size, gen, dev, dt, a.windows, a.steps,
                           guidance=g)
            img[idx] = S.to_uint8(out)[:, 0].cpu().numpy()
        extra = []
        for j in range(b):
            if a.recipe != "v0.1": img[j], e = v3.paint_after(img[j], lays[j], a.seed, k + j, v3.RECIPE_S5)
            else: e = dict(inclusions=0, debris=0)
            extra.append(e)
        np.savez_compressed(os.path.join(a.out, f"shard_{shard:04d}.npz"), image=img, lines=np.stack([x["lines"] for x in lays]),
                            sheet_id=np.stack([x["sheet_id"] for x in lays]), dist=np.stack([x["dist"] for x in lays]), style=np.array(sty),
                            seed=a.seed, kind=np.array([x["kind"] for x in lays]), crack=np.stack([x["crack"] for x in lays]),
                            crushed=np.array([x["crushed"] for x in lays], np.uint8), guidance=np.array([x["guidance"] for x in lays], np.float32),
                            inclusions=np.array([e["inclusions"] for e in extra], np.uint8), debris=np.array([e["debris"] for e in extra], np.uint8))
        if a.png:
            import cv2
            for j in range(b):
                n = k + j; cv2.imwrite(os.path.join(a.out, "png", f"{n:06d}.png"), img[j]); c = np.repeat(img[j][..., None], 3, -1)
                ids = lays[j]["sheet_id"]
                for f in np.unique(ids[ids > 0]): c[ids == f] = PAL[int(f) % len(PAL)]
                cv2.imwrite(os.path.join(a.out, "png", f"{n:06d}_lines.png"), c[..., ::-1])
        k += b; shard += 1; el = time.time() - t0
        print(f"{k}/{a.n} tiles · {el / k:.1f} s per tile · {el:.0f} s · kinds: {', '.join(x['kind'] for x in lays)}", flush=True)
    print(f"done: {a.n} tiles in {a.out}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------- detect
def _read(path, index):
    p = path.lower(); labels = None
    if p.endswith(".npz"):
        z = np.load(path); key = "image" if "image" in z.files else z.files[0]; X = z[key]
        if X.ndim == 3: X = X[index]
        if "lines" in z.files: labels = z["lines"][index] if z["lines"].ndim == 3 else z["lines"]
    elif p.endswith(".npy"):
        X = np.load(path)
    elif p.endswith((".tif", ".tiff")):
        try:
            import tifffile; X = tifffile.imread(path)
        except ImportError:
            import cv2; X = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        import cv2; X = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if X is None: sys.exit(f"cannot read {path}")
        if X.ndim == 3: X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    X = np.asarray(X).squeeze()
    if X.ndim != 2: sys.exit(f"expected a 2D slice, got shape {X.shape} (use --index for a stack in a .npz)")
    return X.astype(np.float32), labels


def _f1(pred, true, tol=2.0):
    from scipy import ndimage as ndi
    P, T = pred > 0, true > 0
    if not P.any() or not T.any(): return 0.0, 0.0, 0.0
    prec = float((ndi.distance_transform_edt(~T)[P] <= tol).mean()); rec = float((ndi.distance_transform_edt(~P)[T] <= tol).mean())
    return prec, rec, 2 * prec * rec / (prec + rec + 1e-9)


def cmd_detect(a):
    import torch, cv2
    from scipy import ndimage as ndi
    from . import detector, extract, hub
    if a.threads: torch.set_num_threads(a.threads)
    X, labels = _read(a.input, a.index); H0, W0 = X.shape
    m = None
    if a.mask:
        m = np.load(a.mask) if a.mask.endswith(".npy") else cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        m = np.asarray(m) > 0
    zoom = a.voxel_um / 9.4
    Xz = ndi.zoom(X, zoom, order=1) if abs(zoom - 1) > 0.02 else X
    w = hub.resolve(a.weights, hub.DETECTORS); dev = a.device
    t0 = time.time(); net = detector.load_detector(w, dev); prob = detector.read_slice(net, Xz, dev); t1 = time.time()
    if Xz is not X: prob = ndi.zoom(prob, (H0 / prob.shape[0], W0 / prob.shape[1]), order=1)[:H0, :W0]
    lines = extract.extract_lines(prob, m); t2 = time.time(); topo = extract.topology(lines)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.input))[0] + (f"_{a.index}" if a.input.lower().endswith(".npz") else "")
    base = os.path.join(a.out, stem)
    cv2.imwrite(base + "_lines.png", lines)
    cv2.imwrite(base + "_prob.png", np.clip(prob * 255, 0, 255).astype(np.uint8))
    lo, hi = np.percentile(X, (1, 99.5)); g = np.clip((X - lo) / (hi - lo + 1e-6), 0, 1) * 0.7
    ov = np.repeat((g * 255).astype(np.uint8)[..., None], 3, -1); ov[ndi.binary_dilation(lines > 0)] = (66, 228, 240)   # BGR = Okabe-Ito yellow
    cv2.imwrite(base + "_overlay.png", ov)
    np.savez_compressed(base + ".npz", prob=prob.astype(np.float16), lines=(lines > 0).astype(np.uint8), weights=os.path.basename(w), voxel_um=a.voxel_um)
    n = max(1, topo["pixels"])
    print(f"{a.input}: {H0} x {W0} px · detector {t1 - t0:.1f} s · NMS 2 {t2 - t1:.1f} s · device {dev}")
    print(f"lines: {topo['pixels']} px in {topo['pieces']} pieces · junctions {topo['junctions']} · line ends {topo['ends']} "
          f"({1000 * topo['ends'] / n:.1f} per 1,000 px)")
    if labels is not None:
        pr, rc, f = _f1(lines, labels)
        print(f"exact labels found in the input: precision {pr:.3f} · recall {rc:.3f} · F1 {f:.3f} at 2 px (whole tile; quick check, not the benchmark metric)")
    print(f"written: {base}_lines.png, _prob.png, _overlay.png, .npz")


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="synthetic-slices", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"synthetic-slices {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    try:
        import torch; cuda = torch.cuda.is_available()
    except Exception: cuda = False

    g = sub.add_parser("generate", help="synthetic tiles + exact labels", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g.add_argument("--n", type=int, default=16, help="number of tiles"); g.add_argument("--size", type=int, default=256, help="tile size (multiple of 128, >= 256)")
    g.add_argument("--out", default="tiles"); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--recipe", default="v0.2", choices=["v0.2", "v0.1"], help="v0.2 = folds, cracks, crushed, edges, inclusions (benchmark recipe); v0.1 = plain stacks")
    g.add_argument("--style", default="random", help="texture style: random, unknown, or an index / name from synthetic_slices.STYLES")
    g.add_argument("--weights", default="v0.2", help="v0.2 (trained without Paris 4), v0.1, or a local .safetensors / .pt file")
    g.add_argument("--batch", type=int, default=0, help="tiles denoised together (default 16 at 256 px on GPU, 4 on CPU or at larger sizes)")
    g.add_argument("--windows", type=int, default=9, help="256 px windows per network call"); g.add_argument("--steps", type=int, default=32, help="sampling steps (32 = released setting)")
    g.add_argument("--margin", type=int, default=48); g.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    g.add_argument("--device", default="cuda" if cuda else "cpu"); g.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    g.add_argument("--png", action="store_true", help="also write PNG previews"); g.set_defaults(fn=cmd_generate)

    d = sub.add_parser("detect", help="sheet lines on a CT slice (no branches)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d.add_argument("input", help="2D slice: .png / .jpg / .tif / .npy, or a .npz from `generate` (key image, tile --index)")
    d.add_argument("--out", default="detect_out", help="output folder"); d.add_argument("--index", type=int, default=0, help="tile index in a .npz stack")
    d.add_argument("--weights", default="seed1", help="seed1 / seed2 / seed3, or a local .safetensors / .pt file")
    d.add_argument("--voxel-um", type=float, default=9.4, help="voxel size of the input; the detector works at ~9.4 um/px and the slice is resampled")
    d.add_argument("--mask", default=None, help="optional .npy / .png mask (non-zero = area to extract)")
    d.add_argument("--device", default="cuda" if cuda else "cpu"); d.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    d.set_defaults(fn=cmd_detect)
    a = ap.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()
