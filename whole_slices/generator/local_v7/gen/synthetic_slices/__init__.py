"""Synthetic-Slices: synthetic Herculaneum scroll CT slices with exact per-sheet labels, and a sheet-line detector trained on them."""
from .model import UNet, STYLES, NS, condition, denoise, to_uint8, load
from .sampler import sample, pick_dtype
from .layout import random_settings, sheet_layout

__version__ = "0.2.0"
