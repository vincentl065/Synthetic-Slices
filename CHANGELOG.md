# Changelog

## v0.3.0

### Added
- **Whole scroll slices** (`whole_slices/generator/`): the shape of a complete cross-section is computed as a winding field
  (loose or crushed scroll, soft folds, cracks, inclusions), the diffusion generator paints it window by window, and a finishing
  step adds motion blur, the tone of a real scroll, inclusions and the edge.
  - `lot.py` makes a batch on one or several GPUs and resumes where it stopped.
  - Cracks are made by breaking the trace given to the painter; the ground truth stays continuous.
  - Faster whole-slice sampler (`sampler_rapide.py`): windows outside the scroll are skipped, 192 px stride, checked against the
    128 px reference.
- **Dataset** `vincentl065/synthetic-scroll-slices` on Hugging Face: 500 slices with their ground truth, the painter weights
  (`generator/`) and the baseline weights (`baseline/`).
- **Baseline tracer** (`whole_slices/baseline_tracer/`): training, data preparation and inference code of a first model trained on
  the dataset, and its scores on held-out synthetic slices and on Paris 4.

## v0.2.0

### Added
- **Command-line tool and pip package** (`pyproject.toml`):
  - `synthetic-slices generate` makes tiles with exact labels; `synthetic-slices detect` gives sheet lines on a slice with no branches.
  - Also runs as `python -m synthetic_slices`.
  - Weights are downloaded from Hugging Face by short name (`synthetic_slices/hub.py`), or read from a local file.
- **v0.2 layouts** (`folds.py`, `hard_cases.py`, `cracks.py`, `phenomena.py`, `v3.py`):
  - shapes: hairpins, tight U folds, tight bends, accordion folds (optionally compact), realistic cracks (sheets cut and offset), mush,
    delamination, scroll edges with frayed last sheets, sheets pressed into contact;
  - painted after the model, without moving any line: the crushed look, the crack opening, mineral inclusions and debris;
  - `v3.RECIPE_S5` is the recipe used for the benchmark training set.
- **Classifier-free guidance** in `sampler.sample` (`guidance=`; 1.0 = off, as in v0.1). v0.2 uses 1.6 for folds and hard cases.
- **Sheet-line detector trained on synthetic tiles only** (`detector.py`) and **NMS 2 line extraction** (`extract.py`).
  - NMS 2 gives one continuous line per sheet, with no junctions and no spurs.
  - It also counts junctions and line ends (`extract.topology`).
- **New outputs** in the `generate` shards: `kind`, `crack`, `crushed`, `guidance`, `inclusions`, `debris`.
- **Benchmark** in the README: synthetic vs real training data vs a classic ridge filter, on Paris 4 and on held-out hand-traced
  slices, with 3 seeds.
- **New weights on Hugging Face:**
  - `generator_v02_noP4_ema_5492.safetensors`: same recipe as v0.1, retrained without any Paris 4 data;
  - `detector_v02_S5_seed{1,2,3}.safetensors`.

### Changed
- **Defaults of `generate`** (and of `generate.py`, now a thin wrapper around the CLI): the v0.2 recipe and the v0.2 generator.
  `--recipe v0.1 --weights v0.1` gives the v0.1 behaviour.
- **Distance condition:** it is now passed at quarter-pixel precision, as in training (v0.1's `generate.py` rounded it to whole
  pixels).
- **OpenCV dependency:** `opencv-python-headless` replaces `opencv-python`.
- **On CPU**, the default batch is 4 tiles.

### Unchanged
- `layout.py`, `model.py`, the v0.1 weights (`synthetic_slices_ema_6655.safetensors`, still available as `--weights v0.1`),
  `training/`, and the licences (code MIT; weights and generated data CC BY-NC 4.0).

## v0.1.0
- First release: procedural layouts with exact per-sheet labels, EDM diffusion texture model, seamless canvases, `generate.py`,
  training code.
