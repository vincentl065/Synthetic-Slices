"""Weights: local path, or a short name downloaded once from the Hugging Face repo (cached by huggingface_hub)."""
import os

HF_REPO = os.environ.get("SYNTH_SLICES_HF", "vincentl065/Synthetic-Slices")

GENERATORS = {
    # v0.2 default: same recipe as v0.1, retrained WITHOUT any Paris 4 data, so Paris 4 stays a fair test (it painted the benchmark tiles)
    "v0.2": "generator_v02_noP4_ema_5492.safetensors",
    # v0.1 release (its training data included Paris 4)
    "v0.1": "synthetic_slices_ema_6655.safetensors",
}
DETECTORS = {
    # sheet-line detector trained on synthetic tiles only (benchmark arm "S5"), 3 training seeds; seed1 is the default
    "seed1": "detector_v02_S5_seed1.safetensors",
    "seed2": "detector_v02_S5_seed2.safetensors",
    "seed3": "detector_v02_S5_seed3.safetensors",
}


def resolve(name, table):
    """local file -> itself; short name (see table) or a file name in the HF repo -> downloaded path"""
    if name and os.path.isfile(name): return name
    fn = table.get(name, name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, fn)
