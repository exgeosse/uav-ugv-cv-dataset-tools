#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Sim-to-Real Augmentation GUI - Final Clean Version

Background Replacement has been removed on purpose.

Main purpose:
- Label-safe Sim-to-Real augmentation for YOLO datasets.
- Auto Smart Strength based on largest YOLO bbox coverage.
- Class-aware augmentation.
- Preset probability logic.
- Quality filter.
- Preview.
- Dataset analysis.
- augmentation_log.csv.

Install:
    pip install pillow numpy pyyaml

Run:
    python yolo_sim2real_gui_final_no_background.py
"""

from __future__ import annotations

import csv
import random
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageEnhance

try:
    import yaml
except Exception:
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "val", "test"]

# Annotation editor display settings
EDITOR_MAX_DISPLAY_SIZE = 760
BOX_COLOR = "red"
SELECTED_BOX_COLOR = "lime"
BOX_WIDTH = 3
HANDLE_SIZE = 8
CROSSHAIR_COLOR = "#d9d9d9"
CROSSHAIR_DASH = (6, 4)


@dataclass
class Preset:
    name: str
    probability: float
    exposure: Tuple[float, float]
    gamma: Tuple[float, float]
    contrast: Tuple[float, float]
    saturation: Tuple[float, float]
    wb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    haze_prob: float
    haze: Tuple[float, float]
    dust_prob: float
    dust: Tuple[float, float]
    fog_prob: float
    fog: Tuple[float, float]
    noise: Tuple[float, float]
    blur_prob: float
    blur: Tuple[float, float]
    motion_prob: float
    vignette_prob: float
    vignette: Tuple[float, float]
    chromatic_prob: float
    jpeg: Tuple[int, int]


PRESETS = [
    Preset("realistic_neutral_camera", 12, (0.82,1.15), (0.90,1.12), (0.85,1.10), (0.70,1.00), ((0.94,1.08),(0.96,1.05),(0.92,1.08)), .30,(.02,.10), .10,(.01,.06), .05,(.01,.05), (4,13), .20,(.20,.70), .04, .25,(.05,.18), .10, (62,90)),
    Preset("overcast_flat_light", 9, (0.82,1.02), (0.95,1.12), (0.82,1.00), (0.65,0.90), ((0.94,1.03),(0.96,1.04),(0.98,1.10)), .35,(.04,.12), .10,(.02,.07), .20,(.03,.10), (5,14), .25,(.25,.75), .05, .25,(.06,.18), .15, (60,88)),
    Preset("sunny_dry_haze", 7, (0.95,1.25), (0.85,1.05), (0.95,1.18), (0.80,1.05), ((1.02,1.12),(0.98,1.06),(0.88,1.02)), .75,(.08,.22), .35,(.04,.13), .05,(.02,.06), (4,11), .18,(.20,.65), .05, .30,(.08,.22), .10, (58,86)),
    Preset("dusty_field_camera", 7, (0.80,1.15), (0.90,1.15), (0.80,1.08), (0.65,0.95), ((1.02,1.18),(0.95,1.06),(0.78,0.98)), .45,(.05,.14), .85,(.08,.24), .05,(.01,.05), (6,18), .30,(.30,.90), .15, .45,(.10,.28), .20, (50,82)),
    Preset("uav_camera_medium_altitude", 8, (0.78,1.18), (0.88,1.14), (0.78,1.12), (0.70,1.00), ((0.94,1.08),(0.96,1.05),(0.92,1.08)), .55,(.05,.17), .25,(.03,.10), .15,(.03,.10), (5,16), .40,(.35,1.00), .12, .35,(.08,.24), .25, (52,84)),
    Preset("cheap_cctv_compressed", 5, (0.72,1.20), (0.85,1.22), (0.75,1.20), (0.55,0.88), ((0.90,1.12),(0.92,1.08),(0.88,1.12)), .30,(.03,.11), .10,(.02,.06), .05,(.01,.04), (10,26), .45,(.40,1.20), .20, .50,(.10,.30), .35, (35,65)),
    Preset("dashcam_low_bitrate", 5, (0.76,1.28), (0.82,1.18), (0.82,1.18), (0.65,1.00), ((0.95,1.14),(0.95,1.05),(0.85,1.10)), .35,(.04,.13), .25,(.03,.11), .05,(.01,.05), (8,22), .35,(.30,1.10), .30, .40,(.08,.25), .30, (38,70)),
    Preset("dusk_low_light_noisy", 4, (0.50,0.85), (1.02,1.35), (0.72,1.05), (0.55,0.85), ((0.92,1.08),(0.90,1.04),(1.02,1.22)), .30,(.03,.11), .05,(.01,.04), .10,(.02,.08), (16,36), .35,(.30,1.10), .15, .55,(.12,.35), .20, (45,75)),
    Preset("early_morning_cool", 5, (0.65,1.00), (0.95,1.20), (0.75,1.05), (0.65,0.95), ((0.86,1.00),(0.94,1.04),(1.05,1.24)), .45,(.04,.14), .05,(.01,.05), .30,(.04,.13), (6,18), .30,(.25,.90), .06, .35,(.08,.25), .15, (55,86)),
    Preset("afternoon_warm", 5, (0.88,1.22), (0.85,1.08), (0.88,1.18), (0.75,1.05), ((1.05,1.20),(0.96,1.08),(0.80,0.98)), .35,(.03,.12), .20,(.02,.08), .02,(.01,.03), (4,13), .18,(.20,.70), .04, .30,(.07,.22), .10, (58,88)),
    Preset("light_fog_surveillance", 4, (0.72,1.05), (0.95,1.18), (0.65,0.95), (0.50,0.82), ((0.94,1.06),(0.96,1.04),(0.98,1.12)), .55,(.06,.18), .02,(.01,.04), .75,(.08,.23), (6,18), .35,(.30,1.00), .05, .25,(.05,.18), .15, (52,82)),
    Preset("cloud_shadow_patchy", 6, (0.70,1.18), (0.88,1.18), (0.78,1.15), (0.65,0.95), ((0.92,1.10),(0.94,1.06),(0.90,1.10)), .40,(.04,.14), .15,(.02,.08), .10,(.02,.08), (6,16), .25,(.25,.85), .08, .35,(.07,.22), .18, (55,84)),
    Preset("backlight_glare", 3, (0.75,1.30), (0.80,1.12), (0.70,1.08), (0.60,0.95), ((1.02,1.18),(0.96,1.06),(0.86,1.04)), .60,(.08,.22), .15,(.03,.10), .05,(.02,.06), (5,16), .25,(.25,.85), .05, .45,(.12,.32), .20, (50,82)),
    Preset("military_field_gritty", 8, (0.68,1.08), (0.92,1.22), (0.75,1.05), (0.45,0.78), ((0.94,1.12),(0.94,1.06),(0.85,1.05)), .55,(.06,.18), .40,(.05,.16), .10,(.02,.08), (10,28), .35,(.35,1.05), .15, .60,(.12,.35), .30, (42,75)),
    Preset("washed_out_surveillance", 4, (0.85,1.35), (0.78,1.05), (0.55,0.88), (0.45,0.75), ((0.94,1.08),(0.96,1.05),(0.95,1.12)), .65,(.08,.24), .15,(.02,.08), .20,(.03,.11), (8,22), .30,(.25,1.00), .10, .25,(.05,.18), .25, (45,76)),
    Preset("high_iso_field_camera", 4, (0.60,1.08), (0.92,1.24), (0.75,1.12), (0.55,0.90), ((0.90,1.12),(0.94,1.06),(0.90,1.14)), .35,(.03,.13), .15,(.02,.08), .05,(.01,.06), (18,40), .35,(.25,.95), .15, .50,(.10,.32), .25, (45,80)),
    Preset("lens_soft_real_camera", 4, (0.75,1.15), (0.88,1.14), (0.75,1.08), (0.60,0.95), ((0.92,1.10),(0.95,1.06),(0.90,1.12)), .35,(.03,.12), .10,(.02,.07), .08,(.02,.07), (5,15), .75,(.50,1.35), .05, .45,(.10,.30), .25, (52,82)),
    Preset("heavy_jpeg_artifacts", 3, (0.72,1.22), (0.85,1.22), (0.70,1.16), (0.50,0.90), ((0.90,1.14),(0.92,1.08),(0.86,1.14)), .30,(.03,.10), .15,(.02,.08), .05,(.01,.05), (8,24), .35,(.25,.95), .20, .40,(.10,.28), .35, (30,58)),
    Preset("dry_summer_humidity", 4, (0.88,1.22), (0.84,1.08), (0.80,1.08), (0.65,0.92), ((1.02,1.16),(0.96,1.06),(0.86,1.02)), .70,(.08,.22), .22,(.03,.10), .05,(.01,.05), (5,16), .20,(.20,.75), .06, .35,(.08,.22), .15, (55,84)),
    Preset("winter_desaturated", 3, (0.70,1.08), (0.96,1.20), (0.70,1.00), (0.42,0.70), ((0.90,1.04),(0.95,1.05),(1.02,1.18)), .40,(.04,.14), .04,(.01,.04), .20,(.03,.12), (6,18), .25,(.25,.85), .05, .30,(.06,.20), .15, (55,85)),
]


def clamp(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def find_yaml(dataset: Path) -> Optional[Path]:
    for name in ["data.yaml", "data.yml"]:
        p = dataset / name
        if p.exists():
            return p
    return None


def load_class_names(dataset: Path) -> Dict[int, str]:
    if yaml is None:
        return {}
    p = find_yaml(dataset)
    if not p:
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
    except Exception:
        return {}
    return {}


def find_splits(dataset: Path):
    splits = []
    for split in SPLITS:
        d = dataset / split
        if not d.exists():
            continue
        img_dir = d / "images" if (d / "images").exists() else d
        lbl_dir = d / "labels" if (d / "labels").exists() else d
        if any(p.suffix.lower() in IMAGE_EXTS for p in img_dir.rglob("*") if p.is_file()):
            splits.append((split, img_dir, lbl_dir))
    if not splits and any(p.suffix.lower() in IMAGE_EXTS for p in dataset.rglob("*") if p.is_file()):
        splits.append(("dataset", dataset, dataset))
    return splits


def images_in(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_for_image(img: Path):
    parts = list(img.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def read_yolo_objects(label: Path):
    objs = []
    if not label.exists():
        return objs
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                objs.append({
                    "class": int(float(parts[0])),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
            except Exception:
                pass
    return objs


def read_classes(label: Path):
    return [o["class"] for o in read_yolo_objects(label)]


def label_coverage_stats(label: Path):
    objs = read_yolo_objects(label)
    areas = [max(0, min(1, o["w"])) * max(0, min(1, o["h"])) for o in objs]
    if not areas:
        return 0.0, 0.0, 0
    return max(areas), min(1.0, sum(areas)), len(areas)


def auto_strength_from_coverage(max_coverage: float, object_count: int) -> str:
    if object_count >= 10 and max_coverage < 0.08:
        return "Light"
    if object_count >= 5 and max_coverage < 0.05:
        return "Light"
    if max_coverage < 0.05:
        return "Light"
    if max_coverage < 0.15:
        return "Medium"
    return "Heavy"


def preset_allowed_for_strength(preset_name: str, selected_strength: str) -> bool:
    aggressive = {
        "cheap_cctv_compressed", "dashcam_low_bitrate", "dusk_low_light_noisy",
        "heavy_jpeg_artifacts", "high_iso_field_camera", "lens_soft_real_camera",
        "light_fog_surveillance",
    }
    medium_risky = {
        "backlight_glare", "washed_out_surveillance", "dusty_field_camera",
        "military_field_gritty",
    }
    if selected_strength == "Light":
        return preset_name not in aggressive and preset_name not in medium_risky
    if selected_strength == "Medium":
        return preset_name not in {"heavy_jpeg_artifacts", "dusk_low_light_noisy"}
    return True


def weighted_choice(py_rng: random.Random, candidates=None):
    candidates = candidates or PRESETS
    total = sum(max(0, p.probability) for p in candidates)
    if total <= 0:
        return candidates[0]
    x = py_rng.uniform(0, total)
    s = 0
    for p in candidates:
        s += max(0, p.probability)
        if s >= x:
            return p
    return candidates[-1]


def weighted_choice_for_strength(py_rng: random.Random, selected_strength: str):
    candidates = [p for p in PRESETS if preset_allowed_for_strength(p.name, selected_strength)]
    return weighted_choice(py_rng, candidates or PRESETS)


def add_atmosphere(arr, strength, mode, np_rng):
    h, w = arr.shape[:2]
    color = np.array([185, 190, 190], dtype=np.float32)
    if mode == "dust":
        color = np.array([190, 170, 125], dtype=np.float32)
    if mode == "fog":
        color = np.array([205, 210, 215], dtype=np.float32)

    y = np.linspace(1.0, 0.35, h).reshape(h, 1, 1)
    alpha = strength * y
    arr = arr * (1 - alpha) + color * alpha

    small = np_rng.normal(0, 1, (max(8, h // 32), max(8, w // 32)))
    small = (small - small.min()) / (np.ptp(small) + 1e-6)
    patch = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    patch = patch.filter(ImageFilter.GaussianBlur(radius=10))
    arr += (np.array(patch).astype(np.float32) / 255.0)[..., None] * strength * 35
    return arr


def vignette(arr, strength):
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    mask = 1 - strength * (dist / dist.max()) ** 1.7
    return arr * mask[..., None]


def chromatic(img, shift):
    a = np.array(img)
    return Image.fromarray(np.stack([
        np.roll(a[:, :, 0], shift, 1),
        a[:, :, 1],
        np.roll(a[:, :, 2], -shift, 1)
    ], axis=2).astype(np.uint8))


def motion_blur(img, horizontal=True):
    if horizontal:
        k = [0,0,0,0,0, 0,0,0,0,0, .2,.2,.2,.2,.2, 0,0,0,0,0, 0,0,0,0,0]
    else:
        k = [0,0,.2,0,0, 0,0,.2,0,0, 0,0,.2,0,0, 0,0,.2,0,0, 0,0,.2,0,0]
    return img.filter(ImageFilter.Kernel((5, 5), k, scale=1))


def lap_var(img):
    g = np.array(img.convert("L")).astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def quality_ok(img, enabled, min_b, max_b, min_sharp):
    if not enabled:
        return True, "disabled", 0, 0
    g = np.array(img.convert("L")).astype(np.float32)
    mean = float(g.mean())
    sharp = lap_var(img)
    if mean < min_b:
        return False, "too_dark", mean, sharp
    if mean > max_b:
        return False, "too_bright", mean, sharp
    if sharp < min_sharp:
        return False, "too_blurry", mean, sharp
    return True, "ok", mean, sharp


def augment(img, preset, strength, py_rng, np_rng):
    scale = {"Light": .55, "Medium": 1.0, "Heavy": 1.35}.get(
        strength, py_rng.choice([.55, .8, 1.0, 1.2, 1.35])
    )

    def neutral(v):
        return 1 + (v - 1) * scale

    arr = np.array(img.convert("RGB")).astype(np.float32)

    exposure = neutral(py_rng.uniform(*preset.exposure))
    gamma = neutral(py_rng.uniform(*preset.gamma))
    contrast = neutral(py_rng.uniform(*preset.contrast))
    saturation = neutral(py_rng.uniform(*preset.saturation))

    arr *= exposure
    arr = 255 * ((np.clip(arr, 0, 255) / 255) ** gamma)

    pil = Image.fromarray(clamp(arr))
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Color(pil).enhance(saturation)
    arr = np.array(pil).astype(np.float32)

    arr[:, :, 0] *= neutral(py_rng.uniform(*preset.wb[0]))
    arr[:, :, 1] *= neutral(py_rng.uniform(*preset.wb[1]))
    arr[:, :, 2] *= neutral(py_rng.uniform(*preset.wb[2]))

    weather = []

    if py_rng.random() < preset.haze_prob:
        s = py_rng.uniform(*preset.haze) * scale
        arr = add_atmosphere(arr, s, "haze", np_rng)
        weather.append(f"haze:{s:.3f}")

    if py_rng.random() < preset.dust_prob:
        s = py_rng.uniform(*preset.dust) * scale
        arr = add_atmosphere(arr, s, "dust", np_rng)
        weather.append(f"dust:{s:.3f}")

    if py_rng.random() < preset.fog_prob:
        s = py_rng.uniform(*preset.fog) * scale
        arr = add_atmosphere(arr, s, "fog", np_rng)
        weather.append(f"fog:{s:.3f}")

    if py_rng.random() < .35 * min(1.0, scale):
        h, w = arr.shape[:2]
        grad = np.linspace(py_rng.uniform(.75, 1.05), py_rng.uniform(.65, 1.05), w).reshape(1, w, 1)
        arr *= grad
        weather.append("cloud_shadow")

    sigma = py_rng.uniform(*preset.noise) * scale
    arr += np_rng.normal(0, sigma, arr.shape[:2])[..., None]
    arr += np_rng.normal(0, sigma * .35, arr.shape)

    vig = 0
    if py_rng.random() < preset.vignette_prob:
        vig = py_rng.uniform(*preset.vignette) * scale
        arr = vignette(arr, vig)

    pil = Image.fromarray(clamp(arr))

    blur = "none"
    if py_rng.random() < preset.motion_prob * min(1.0, scale):
        pil = motion_blur(pil, py_rng.choice([True, False]))
        blur = "motion"
    elif py_rng.random() < preset.blur_prob:
        r = py_rng.uniform(*preset.blur) * scale
        pil = pil.filter(ImageFilter.GaussianBlur(radius=max(.05, r)))
        blur = f"gaussian:{r:.2f}"

    chrom = 0
    if py_rng.random() < preset.chromatic_prob:
        chrom = py_rng.choice([1, 2])
        pil = chromatic(pil, chrom)

    sharpness = py_rng.uniform(.8, 1.2)
    pil = ImageEnhance.Sharpness(pil).enhance(sharpness)

    jpeg = py_rng.randint(*preset.jpeg)

    params = {
        "preset": preset.name,
        "selected_strength": strength,
        "strength_scale": f"{scale:.2f}",
        "exposure": f"{exposure:.3f}",
        "gamma": f"{gamma:.3f}",
        "contrast": f"{contrast:.3f}",
        "saturation": f"{saturation:.3f}",
        "weather": "|".join(weather) or "none",
        "noise_sigma": f"{sigma:.2f}",
        "vignette": f"{vig:.3f}",
        "blur": blur,
        "chromatic_shift": str(chrom),
        "sharpness": f"{sharpness:.3f}",
        "jpeg_quality": str(jpeg),
    }

    return pil, params


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Sim-to-Real GUI - Final No Background Replacement")
        self.geometry("1200x780")

        self.dataset = tk.StringVar(value=str(DATASET_DIR))
        self.output = tk.StringVar(value=str(DATASET_DIR / "dataset_sim2real_output"))

        self.seed = tk.StringVar(value="42")
        self.factor = tk.IntVar(value=1)
        self.strength = tk.StringVar(value="Auto")
        self.keep_original = tk.BooleanVar(value=False)
        self.quality_filter = tk.BooleanVar(value=True)
        self.class_aware = tk.BooleanVar(value=False)

        self.min_b = tk.DoubleVar(value=32.0)
        self.max_b = tk.DoubleVar(value=232.0)
        self.min_sharp = tk.DoubleVar(value=18.0)

        self.progress = tk.DoubleVar(value=0)
        self.status = tk.StringVar(value="Ready. Background Replacement removed from this version.")
        self.preview_path = None
        self.low_value_images = []
        self.low_index = 0
        self.low_threshold = tk.DoubleVar(value=1.0)  # percent
        self.low_imgtk = None
        self.low_info = tk.StringVar(value="No low-value images loaded.")

        # Integrated annotation editor state for Tab 4
        self.editor_original_image = None
        self.editor_display_image = None
        self.editor_tk_image = None
        self.editor_annotations = []
        self.editor_selected = set()
        self.editor_current_image = None
        self.editor_current_label = None
        self.editor_scale = 1.0
        self.editor_action = None
        self.editor_resize_handle = None
        self.editor_start_x = None
        self.editor_start_y = None
        self.editor_drag_last_x = None
        self.editor_drag_last_y = None
        self.editor_drawing = False
        self.editor_temp_box = None
        self.editor_crosshair_items = []
        self.orig_imgtk = None
        self.aug_imgtk = None
        self.class_names = {}

        self.build()

    def build(self):
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        paths = ttk.LabelFrame(main, text="Dataset Paths", padding=8)
        paths.pack(fill="x")

        ttk.Label(paths, text="Input dataset folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(paths, textvariable=self.dataset).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(paths, text="Browse", command=self.browse_dataset).grid(row=0, column=2)

        ttk.Label(paths, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(paths, textvariable=self.output).grid(row=1, column=1, sticky="ew", padx=5, pady=(6, 0))
        ttk.Button(paths, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=(6, 0))

        paths.columnconfigure(1, weight=1)

        nb = ttk.Notebook(main)
        nb.pack(fill="both", expand=True, pady=8)

        self.tab_aug = ttk.Frame(nb, padding=10)
        self.tab_analysis = ttk.Frame(nb, padding=10)
        self.tab_preview = ttk.Frame(nb, padding=10)
        self.tab_low_value = ttk.Frame(nb, padding=10)

        nb.add(self.tab_aug, text="Tab 1: Sim-to-Real Augmentation")
        nb.add(self.tab_analysis, text="Tab 2: Dataset Analysis")
        nb.add(self.tab_preview, text="Tab 3: Preview & Statistics")
        nb.add(self.tab_low_value, text="Tab 4: Low Value Review")

        self.build_aug_tab()
        self.build_analysis_tab()
        self.build_preview_tab()
        self.build_low_value_tab()

        ttk.Progressbar(main, variable=self.progress, maximum=100).pack(fill="x")
        ttk.Label(main, textvariable=self.status).pack(anchor="w", pady=(5, 0))

    def build_aug_tab(self):
        left = ttk.LabelFrame(self.tab_aug, text="Final Sim-to-Real Settings", padding=10)
        left.pack(side="left", fill="y")

        ttk.Checkbutton(left, text="Label-safe mode (always on)", state="disabled").pack(anchor="w")
        ttk.Checkbutton(left, text="Class-aware augmentation", variable=self.class_aware).pack(anchor="w")
        ttk.Checkbutton(left, text="Quality filter", variable=self.quality_filter).pack(anchor="w")
        ttk.Checkbutton(left, text="Keep original images in output", variable=self.keep_original).pack(anchor="w")

        ttk.Separator(left).pack(fill="x", pady=8)

        ttk.Label(left, text="Augmentation factor per image:").pack(anchor="w")
        ttk.Spinbox(left, from_=1, to=10, textvariable=self.factor, width=8).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Strength:").pack(anchor="w")
        ttk.Combobox(left, values=["Auto", "Light", "Medium", "Heavy", "Mixed"], textvariable=self.strength, state="readonly", width=12).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Seed:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.seed, width=12).pack(anchor="w", pady=(0, 8))

        ttk.Separator(left).pack(fill="x", pady=8)

        ttk.Label(left, text="Quality thresholds").pack(anchor="w")
        for label, var in [("Min brightness", self.min_b), ("Max brightness", self.max_b), ("Min sharpness", self.min_sharp)]:
            row = ttk.Frame(left)
            row.pack(anchor="w")
            ttk.Label(row, text=label, width=15).pack(side="left")
            ttk.Entry(row, textvariable=var, width=8).pack(side="left")

        ttk.Separator(left).pack(fill="x", pady=8)

        ttk.Label(left, text="Class multipliers").pack(anchor="w")
        self.class_text = tk.Text(left, width=38, height=12)
        self.class_text.pack(fill="x")
        self.class_text.insert("1.0", "# class_id:factor\n# Example:\n# 0:1\n# 1:2\n")

        ttk.Button(left, text="Analyze Dataset Classes", command=self.analyze_classes).pack(fill="x", pady=(8, 3))
        ttk.Button(left, text="Process Dataset", command=self.process_threaded).pack(fill="x", pady=3)

        info = ttk.LabelFrame(self.tab_aug, text="Method", padding=10)
        info.pack(side="left", fill="both", expand=True, padx=(10, 0))

        msg = (
            "Background Replacement has been removed.\n\n"
            "This version applies only label-safe Sim-to-Real augmentation:\n"
            "- exposure variation\n"
            "- white balance / color temperature\n"
            "- haze, dust, light fog\n"
            "- ISO / sensor noise\n"
            "- JPEG artifacts\n"
            "- lens softness / motion blur\n"
            "- vignette and chromatic aberration\n\n"
            "Auto Strength logic:\n"
            "<5% largest bbox coverage  → Light\n"
            "5-15% largest bbox coverage → Medium\n"
            ">15% largest bbox coverage  → Heavy\n\n"
            "This avoids destroying small objects with aggressive blur/noise."
        )
        ttk.Label(info, text=msg, justify="left").pack(anchor="nw")

    def build_analysis_tab(self):
        buttons = ttk.Frame(self.tab_analysis)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Analyze Dataset", command=self.analyze_dataset_full).pack(side="left")
        ttk.Button(buttons, text="Refresh Class Multipliers", command=self.analyze_classes).pack(side="left", padx=5)
        ttk.Label(buttons, text="Low-value threshold (%):").pack(side="left", padx=(20, 4))
        ttk.Entry(buttons, textvariable=self.low_threshold, width=6).pack(side="left")
        ttk.Button(buttons, text="Find Low Value Images", command=self.find_low_value_images).pack(side="left", padx=5)

        self.analysis = tk.Text(self.tab_analysis, wrap="word")
        self.analysis.pack(fill="both", expand=True, pady=8)


    def build_low_value_tab(self):
        top = ttk.Frame(self.tab_low_value)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Low Value Review + Annotation Editor. Review small-object images, fix wrong bboxes/classes, or delete image+label."
        ).pack(side="left")

        controls = ttk.Frame(self.tab_low_value)
        controls.pack(fill="x", pady=6)

        ttk.Label(controls, text="Threshold (%):").pack(side="left")
        ttk.Entry(controls, textvariable=self.low_threshold, width=6).pack(side="left", padx=4)

        ttk.Button(controls, text="Load / Refresh Low Value Images", command=self.find_low_value_images).pack(side="left", padx=4)
        ttk.Button(controls, text="Previous", command=self.low_prev).pack(side="left", padx=(20, 4))
        ttk.Button(controls, text="Next", command=self.low_next).pack(side="left", padx=4)
        ttk.Button(controls, text="Save Annotation", command=self.editor_save_annotations).pack(side="left", padx=(20, 4))
        ttk.Button(controls, text="Save & Next", command=self.editor_save_and_next).pack(side="left", padx=4)
        ttk.Button(controls, text="Delete Current image+label", command=self.delete_current_low_value_pair).pack(side="left", padx=(20, 4))

        class_controls = ttk.Frame(self.tab_low_value)
        class_controls.pack(fill="x", pady=4)

        ttk.Label(class_controls, text="Class:").pack(side="left")
        self.editor_class_var = tk.StringVar()
        self.editor_class_combo = ttk.Combobox(
            class_controls,
            textvariable=self.editor_class_var,
            values=[],
            state="readonly",
            width=38
        )
        self.editor_class_combo.pack(side="left", padx=5)

        ttk.Button(class_controls, text="Change SELECTED", command=self.editor_change_selected_class).pack(side="left", padx=4)
        ttk.Button(class_controls, text="Change ALL", command=self.editor_change_all_classes).pack(side="left", padx=4)
        ttk.Button(class_controls, text="Delete SELECTED Box", command=self.editor_delete_selected_box).pack(side="left", padx=4)

        ttk.Label(self.tab_low_value, textvariable=self.low_info, font=("Arial", 10, "bold")).pack(anchor="w", pady=4)

        self.editor_canvas = tk.Canvas(self.tab_low_value, bg="black", cursor="cross")
        self.editor_canvas.pack(fill="both", expand=True, pady=6)

        self.editor_canvas.bind("<ButtonPress-1>", self.editor_mouse_down)
        self.editor_canvas.bind("<B1-Motion>", self.editor_mouse_drag)
        self.editor_canvas.bind("<ButtonRelease-1>", self.editor_mouse_up)
        self.editor_canvas.bind("<Motion>", self.editor_mouse_move)
        self.editor_canvas.bind("<Leave>", self.editor_clear_crosshair)

        note = (
            "Controls: click box=select | Ctrl+click=multi-select | drag selected box=move | "
            "drag green corner=resize | drag empty area=new box | Save Annotation writes YOLO txt."
        )
        ttk.Label(self.tab_low_value, text=note, justify="left").pack(anchor="w", pady=4)

    def find_low_value_images(self):
        dataset = Path(self.dataset.get())
        threshold = max(0.0, float(self.low_threshold.get())) / 100.0

        self.low_value_images = []

        for split, img_dir, _ in find_splits(dataset):
            for img in images_in(img_dir):
                label = label_for_image(img)
                max_cov, total_cov, obj_count = label_coverage_stats(label)
                if obj_count > 0 and max_cov < threshold:
                    self.low_value_images.append({
                        "split": split,
                        "image": img,
                        "label": label,
                        "max_cov": max_cov,
                        "total_cov": total_cov,
                        "object_count": obj_count,
                    })

        self.low_value_images.sort(key=lambda x: x["max_cov"])
        self.low_index = 0

        if not self.low_value_images:
            self.low_imgtk = None
            self.low_image_label.configure(image="", text="No low-value images found.")
            self.low_info.set(f"No low-value images found below {self.low_threshold.get():.2f}%.")
            self.status.set("Low-value search complete: 0 images.")
            return

        self.show_low_value_image()
        self.status.set(f"Low-value search complete: {len(self.low_value_images)} images.")

    def show_low_value_image(self):
        if not self.low_value_images:
            self.editor_canvas.delete("all")
            self.low_info.set("No low-value images loaded.")
            return

        self.low_index = max(0, min(self.low_index, len(self.low_value_images) - 1))
        item = self.low_value_images[self.low_index]

        self.editor_current_image = item["image"]
        self.editor_current_label = item["label"]

        try:
            self.editor_original_image = Image.open(self.editor_current_image).convert("RGB")
        except Exception as e:
            self.editor_canvas.delete("all")
            self.low_info.set(f"Could not load image: {e}")
            return

        self.editor_annotations = read_yolo_objects(self.editor_current_label)
        # normalize key name expected by editor methods
        for ann in self.editor_annotations:
            if "class_id" not in ann:
                ann["class_id"] = ann.get("class", 0)
        self.editor_selected = set()

        self.update_editor_class_combo()
        self.render_editor_image()

        self.low_info.set(
            f"{self.low_index + 1}/{len(self.low_value_images)} | "
            f"split={item['split']} | "
            f"largest bbox={item['max_cov']*100:.3f}% | "
            f"total bbox={item['total_cov']*100:.3f}% | "
            f"objects={item['object_count']} | "
            f"{self.editor_current_image.name}"
        )

    def update_editor_class_combo(self):
        dataset = Path(self.dataset.get())
        self.class_names = load_class_names(dataset)
        if not self.class_names:
            # fallback from labels/classes in dataset
            ids = set()
            for _, img_dir, _ in find_splits(dataset):
                for img in images_in(img_dir):
                    ids.update(read_classes(label_for_image(img)))
            self.class_names = {i: str(i) for i in sorted(ids)}

        values = [f"{i}: {name}" for i, name in sorted(self.class_names.items())]
        self.editor_class_combo.configure(values=values)

        if values and not self.editor_class_var.get():
            self.editor_class_var.set(values[0])

    def editor_current_class_id(self):
        if not self.editor_class_var.get():
            return 0
        return int(self.editor_class_var.get().split(":")[0])

    def render_editor_image(self):
        if self.editor_original_image is None:
            return

        img = self.editor_original_image.copy()
        w, h = img.size
        scale = min(EDITOR_MAX_DISPLAY_SIZE / w, EDITOR_MAX_DISPLAY_SIZE / h, 1.0)
        self.editor_scale = scale
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        for idx, ann in enumerate(self.editor_annotations):
            x1, y1, x2, y2 = self.editor_annotation_to_canvas_box(ann, new_w, new_h)
            color = SELECTED_BOX_COLOR if idx in self.editor_selected else BOX_COLOR
            draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

            if idx in self.editor_selected:
                hs = HANDLE_SIZE / 2
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    draw.rectangle([hx - hs, hy - hs, hx + hs, hy + hs], fill=color, outline="black")

            cid = ann.get("class_id", ann.get("class", 0))
            name = self.class_names.get(cid, str(cid))
            draw.text((x1, max(0, y1 - 15)), name, fill=color)

        self.editor_display_image = img
        self.editor_tk_image = ImageTk.PhotoImage(img)
        self.editor_canvas.delete("all")
        self.editor_canvas.config(width=new_w, height=new_h)
        self.editor_canvas.create_image(0, 0, anchor=tk.NW, image=self.editor_tk_image)

    def editor_annotation_to_canvas_box(self, ann, img_w=None, img_h=None):
        if img_w is None or img_h is None:
            img_w, img_h = self.editor_display_image.size

        x_center = ann["x"] * img_w
        y_center = ann["y"] * img_h
        box_w = ann["w"] * img_w
        box_h = ann["h"] * img_h

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2
        return x1, y1, x2, y2

    def editor_clamp_point(self, x, y):
        if self.editor_display_image is None:
            return x, y
        w, h = self.editor_display_image.size
        return max(0, min(x, w)), max(0, min(y, h))

    def editor_find_box_at(self, x, y):
        for idx, ann in enumerate(self.editor_annotations):
            x1, y1, x2, y2 = self.editor_annotation_to_canvas_box(ann)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None

    def editor_get_handle_at(self, x, y):
        if len(self.editor_selected) != 1:
            return None, None

        idx = next(iter(self.editor_selected))
        x1, y1, x2, y2 = self.editor_annotation_to_canvas_box(self.editor_annotations[idx])

        handles = {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "sw": (x1, y2),
            "se": (x2, y2),
        }

        for handle, (hx, hy) in handles.items():
            if abs(x - hx) <= HANDLE_SIZE and abs(y - hy) <= HANDLE_SIZE:
                return idx, handle

        return None, None

    def editor_update_annotation_from_canvas_box(self, idx, x1, y1, x2, y2):
        img_w, img_h = self.editor_display_image.size

        x1, y1 = self.editor_clamp_point(x1, y1)
        x2, y2 = self.editor_clamp_point(x2, y2)

        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)

        if right - left < 2 or bottom - top < 2:
            return

        self.editor_annotations[idx]["x"] = ((left + right) / 2) / img_w
        self.editor_annotations[idx]["y"] = ((top + bottom) / 2) / img_h
        self.editor_annotations[idx]["w"] = (right - left) / img_w
        self.editor_annotations[idx]["h"] = (bottom - top) / img_h

    def editor_canvas_to_yolo(self, x1, y1, x2, y2):
        img_w, img_h = self.editor_display_image.size
        return (
            ((x1 + x2) / 2) / img_w,
            ((y1 + y2) / 2) / img_h,
            abs(x2 - x1) / img_w,
            abs(y2 - y1) / img_h,
        )

    def editor_draw_crosshair(self, x, y):
        self.editor_clear_crosshair()
        if self.editor_display_image is None:
            return
        img_w, img_h = self.editor_display_image.size
        x, y = self.editor_clamp_point(x, y)
        self.editor_crosshair_items = [
            self.editor_canvas.create_line(0, y, img_w, y, fill=CROSSHAIR_COLOR, dash=CROSSHAIR_DASH, width=1, tags="crosshair"),
            self.editor_canvas.create_line(x, 0, x, img_h, fill=CROSSHAIR_COLOR, dash=CROSSHAIR_DASH, width=1, tags="crosshair"),
        ]

    def editor_clear_crosshair(self, event=None):
        if hasattr(self, "editor_canvas"):
            self.editor_canvas.delete("crosshair")
        self.editor_crosshair_items = []

    def editor_mouse_move(self, event):
        if self.editor_display_image is None:
            return
        idx, handle = self.editor_get_handle_at(event.x, event.y)
        if handle in ("nw", "se"):
            self.editor_canvas.config(cursor="size_nw_se")
        elif handle in ("ne", "sw"):
            self.editor_canvas.config(cursor="size_ne_sw")
        elif self.editor_find_box_at(event.x, event.y) is not None:
            self.editor_canvas.config(cursor="fleur")
        else:
            self.editor_canvas.config(cursor="cross")
        self.editor_draw_crosshair(event.x, event.y)

    def editor_mouse_down(self, event):
        if self.editor_display_image is None:
            return

        x, y = self.editor_clamp_point(event.x, event.y)
        ctrl_pressed = (event.state & 0x0004) != 0

        handle_idx, handle = self.editor_get_handle_at(x, y)
        if handle_idx is not None:
            self.editor_action = "resize"
            self.editor_resize_handle = handle
            self.editor_drag_last_x = x
            self.editor_drag_last_y = y
            return

        clicked_idx = self.editor_find_box_at(x, y)
        if clicked_idx is not None:
            if ctrl_pressed:
                if clicked_idx in self.editor_selected:
                    self.editor_selected.remove(clicked_idx)
                    self.editor_action = None
                else:
                    self.editor_selected.add(clicked_idx)
                    self.editor_action = "move"
            else:
                self.editor_selected = {clicked_idx}
                self.editor_action = "move"

            self.editor_drag_last_x = x
            self.editor_drag_last_y = y
            self.render_editor_image()
            self.editor_draw_crosshair(x, y)
            return

        if not ctrl_pressed:
            self.editor_selected.clear()

        self.editor_action = "draw"
        self.editor_drawing = True
        self.editor_start_x = x
        self.editor_start_y = y

    def editor_mouse_drag(self, event):
        if self.editor_display_image is None:
            return

        x, y = self.editor_clamp_point(event.x, event.y)

        if self.editor_action == "move" and self.editor_selected:
            dx = x - self.editor_drag_last_x
            dy = y - self.editor_drag_last_y
            img_w, img_h = self.editor_display_image.size

            for idx in self.editor_selected:
                ann = self.editor_annotations[idx]
                box_w, box_h = ann["w"], ann["h"]
                ann["x"] = max(box_w / 2, min(1 - box_w / 2, ann["x"] + dx / img_w))
                ann["y"] = max(box_h / 2, min(1 - box_h / 2, ann["y"] + dy / img_h))

            self.editor_drag_last_x = x
            self.editor_drag_last_y = y
            self.render_editor_image()
            self.editor_draw_crosshair(x, y)
            return

        if self.editor_action == "resize" and len(self.editor_selected) == 1:
            idx = next(iter(self.editor_selected))
            x1, y1, x2, y2 = self.editor_annotation_to_canvas_box(self.editor_annotations[idx])

            if self.editor_resize_handle == "nw":
                x1, y1 = x, y
            elif self.editor_resize_handle == "ne":
                x2, y1 = x, y
            elif self.editor_resize_handle == "sw":
                x1, y2 = x, y
            elif self.editor_resize_handle == "se":
                x2, y2 = x, y

            self.editor_update_annotation_from_canvas_box(idx, x1, y1, x2, y2)
            self.render_editor_image()
            self.editor_draw_crosshair(x, y)
            return

        if self.editor_action != "draw" or not self.editor_drawing:
            return

        self.render_editor_image()
        self.editor_temp_box = self.editor_canvas.create_rectangle(
            self.editor_start_x,
            self.editor_start_y,
            x,
            y,
            outline="yellow",
            width=2
        )
        self.editor_draw_crosshair(x, y)

    def editor_mouse_up(self, event):
        if self.editor_display_image is None:
            return

        x, y = self.editor_clamp_point(event.x, event.y)

        if self.editor_action in ("move", "resize"):
            self.editor_action = None
            self.editor_resize_handle = None
            self.editor_drag_last_x = None
            self.editor_drag_last_y = None
            self.render_editor_image()
            self.editor_draw_crosshair(x, y)
            return

        if not self.editor_drawing:
            self.editor_action = None
            return

        self.editor_drawing = False
        self.editor_action = None

        x1 = min(self.editor_start_x, x)
        y1 = min(self.editor_start_y, y)
        x2 = max(self.editor_start_x, x)
        y2 = max(self.editor_start_y, y)

        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.render_editor_image()
            self.editor_draw_crosshair(x, y)
            return

        x_center, y_center, box_w, box_h = self.editor_canvas_to_yolo(x1, y1, x2, y2)
        class_id = self.editor_current_class_id()

        self.editor_annotations.append({
            "class_id": class_id,
            "x": x_center,
            "y": y_center,
            "w": box_w,
            "h": box_h,
        })

        self.editor_selected = {len(self.editor_annotations) - 1}
        self.render_editor_image()
        self.editor_draw_crosshair(x, y)

    def editor_change_selected_class(self):
        if not self.editor_selected:
            messagebox.showwarning("Warning", "Δεν έχει επιλεγεί κανένα box.")
            return

        class_id = self.editor_current_class_id()
        for idx in self.editor_selected:
            self.editor_annotations[idx]["class_id"] = class_id
        self.render_editor_image()

    def editor_change_all_classes(self):
        class_id = self.editor_current_class_id()
        for ann in self.editor_annotations:
            ann["class_id"] = class_id
        self.render_editor_image()

    def editor_delete_selected_box(self):
        if not self.editor_selected:
            messagebox.showwarning("Warning", "Δεν έχει επιλεγεί κανένα box.")
            return

        for idx in sorted(self.editor_selected, reverse=True):
            del self.editor_annotations[idx]

        self.editor_selected.clear()
        self.render_editor_image()

    def editor_save_annotations(self, show_popup=True):
        if self.editor_current_label is None:
            return

        lines = []
        for ann in self.editor_annotations:
            cid = ann.get("class_id", ann.get("class", 0))
            lines.append(
                f"{cid} "
                f"{ann['x']:.6f} "
                f"{ann['y']:.6f} "
                f"{ann['w']:.6f} "
                f"{ann['h']:.6f}"
            )

        self.editor_current_label.parent.mkdir(parents=True, exist_ok=True)
        self.editor_current_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # refresh current item stats
        if self.low_value_images:
            item = self.low_value_images[self.low_index]
            max_cov, total_cov, obj_count = label_coverage_stats(self.editor_current_label)
            item["max_cov"] = max_cov
            item["total_cov"] = total_cov
            item["object_count"] = obj_count
            self.low_info.set(
                f"{self.low_index + 1}/{len(self.low_value_images)} | "
                f"split={item['split']} | "
                f"largest bbox={max_cov*100:.3f}% | "
                f"total bbox={total_cov*100:.3f}% | "
                f"objects={obj_count} | "
                f"{self.editor_current_image.name}"
            )

        if show_popup:
            messagebox.showinfo("Saved", "Το annotation αποθηκεύτηκε.")

    def editor_save_and_next(self):
        self.editor_save_annotations(show_popup=False)
        self.low_next()

    def low_next(self):
        if not self.low_value_images:
            return
        if self.low_index < len(self.low_value_images) - 1:
            self.low_index += 1
            self.show_low_value_image()
        else:
            messagebox.showinfo("End", "Τελευταία low-value εικόνα.")

    def low_prev(self):
        if not self.low_value_images:
            return
        if self.low_index > 0:
            self.low_index -= 1
            self.show_low_value_image()

    def delete_current_low_value_pair(self):
        if not self.low_value_images:
            return

        item = self.low_value_images[self.low_index]
        img_path = item["image"]
        label_path = item["label"]

        confirm = messagebox.askyesno(
            "Confirm delete",
            f"Θέλεις σίγουρα να διαγράψεις οριστικά το ζεύγος image+label;\\n\\n"
            f"Image:\\n{img_path}\\n\\nLabel:\\n{label_path}"
        )
        if not confirm:
            return

        try:
            if img_path.exists():
                img_path.unlink()
            if label_path.exists():
                label_path.unlink()

            del self.low_value_images[self.low_index]

            if self.low_index >= len(self.low_value_images):
                self.low_index = max(0, len(self.low_value_images) - 1)

            self.show_low_value_image()
            self.status.set(f"Deleted: {img_path.name}")

        except Exception as e:
            messagebox.showerror("Delete error", f"Δεν μπόρεσε να γίνει διαγραφή:\\n{e}")


    def build_preview_tab(self):
        buttons = ttk.Frame(self.tab_preview)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Load Random Image", command=self.load_random).pack(side="left")
        ttk.Button(buttons, text="Preview Sim-to-Real", command=self.preview_s2r).pack(side="left", padx=5)

        imgs = ttk.Frame(self.tab_preview)
        imgs.pack(fill="both", expand=True, pady=8)

        self.orig_label = ttk.Label(imgs, text="Original", anchor="center")
        self.orig_label.pack(side="left", fill="both", expand=True, padx=4)

        self.aug_label = ttk.Label(imgs, text="Augmented", anchor="center")
        self.aug_label.pack(side="left", fill="both", expand=True, padx=4)

    def browse_dataset(self):
        d = filedialog.askdirectory(initialdir=self.dataset.get())
        if d:
            self.dataset.set(d)
            self.output.set(str(Path(d).parent / (Path(d).name + "_sim2real_output")))
            self.class_names = load_class_names(Path(d))

    def browse_output(self):
        d = filedialog.askdirectory(initialdir=self.output.get())
        if d:
            self.output.set(d)

    def all_images(self):
        d = Path(self.dataset.get())
        imgs = []
        for _, img_dir, _ in find_splits(d):
            imgs += images_in(img_dir)
        return imgs

    def fit(self, img, max_size=(500, 500)):
        img = img.copy()
        img.thumbnail(max_size)
        return ImageTk.PhotoImage(img)

    def load_random(self):
        imgs = self.all_images()
        if not imgs:
            messagebox.showerror("No images", "Δεν βρέθηκαν εικόνες.")
            return
        self.preview_path = random.choice(imgs)
        img = Image.open(self.preview_path).convert("RGB")
        self.orig_imgtk = self.fit(img)
        self.orig_label.configure(image=self.orig_imgtk, text="")
        self.status.set(f"Loaded: {self.preview_path.name}")

    def selected_strength_for_label(self, label):
        max_cov, total_cov, obj_count = label_coverage_stats(label)
        mode = self.strength.get()
        selected = auto_strength_from_coverage(max_cov, obj_count) if mode == "Auto" else mode
        return selected, max_cov, total_cov, obj_count

    def preview_s2r(self):
        if self.preview_path is None:
            self.load_random()
        if self.preview_path is None:
            return

        seed = int(self.seed.get() or 42) + random.randint(0, 999999)
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        label = label_for_image(self.preview_path)
        selected, max_cov, _, count = self.selected_strength_for_label(label)

        if self.strength.get() == "Auto":
            preset = weighted_choice_for_strength(py_rng, selected)
        else:
            preset = weighted_choice(py_rng)

        aug, params = augment(Image.open(self.preview_path), preset, selected, py_rng, np_rng)

        self.aug_imgtk = self.fit(aug)
        self.aug_label.configure(image=self.aug_imgtk, text="")

        self.status.set(
            f"Preview: {params['preset']} | strength={selected} | "
            f"largest bbox={max_cov*100:.1f}% | objects={count}"
        )

    def parse_mult(self):
        res = {}
        for line in self.class_text.get("1.0", "end").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                res[int(k.strip())] = max(1, int(v.strip()))
            except Exception:
                pass
        return res

    def analyze_classes(self):
        dataset = Path(self.dataset.get())
        self.class_names = load_class_names(dataset)
        counts = {}
        img_counts = {}

        for _, img_dir, _ in find_splits(dataset):
            for img in images_in(img_dir):
                classes = read_classes(label_for_image(img))
                for c in classes:
                    counts[c] = counts.get(c, 0) + 1
                for c in set(classes):
                    img_counts[c] = img_counts.get(c, 0) + 1

        if not img_counts:
            messagebox.showwarning("No labels", "Δεν βρέθηκαν labels.")
            return

        mn = min(img_counts.values())
        lines = ["# class_id:factor", "# auto suggestion based on image count"]
        for c in sorted(img_counts):
            n = img_counts[c]
            factor = 3 if n <= mn * 1.25 else 2 if n <= mn * 2 else 1
            name = self.class_names.get(c, "")
            lines.append(f"{c}:{factor}  # {name} images={n}, labels={counts.get(c, 0)}")

        self.class_text.delete("1.0", "end")
        self.class_text.insert("1.0", "\n".join(lines))
        self.status.set(f"Found {len(img_counts)} classes.")

    def analyze_dataset_full(self):
        dataset = Path(self.dataset.get())
        self.class_names = load_class_names(dataset)
        splits = find_splits(dataset)

        total_images = 0
        missing_labels = 0
        empty_labels = 0
        class_img_counts = {}
        class_obj_counts = {}
        covs = []
        per_split = {}

        for split, img_dir, _ in splits:
            imgs = images_in(img_dir)
            per_split[split] = len(imgs)
            total_images += len(imgs)

            for img in imgs:
                label = label_for_image(img)
                if not label.exists():
                    missing_labels += 1
                    continue

                objs = read_yolo_objects(label)
                if not objs:
                    empty_labels += 1

                max_cov, _, _ = label_coverage_stats(label)
                covs.append(max_cov)

                for o in objs:
                    c = o["class"]
                    class_obj_counts[c] = class_obj_counts.get(c, 0) + 1

                for c in set(o["class"] for o in objs):
                    class_img_counts[c] = class_img_counts.get(c, 0) + 1

        txt = []
        txt.append("DATASET ANALYSIS\n\n")
        txt.append(f"Dataset: {dataset}\n")
        txt.append(f"Total images: {total_images}\n")
        txt.append(f"Missing labels: {missing_labels}\n")
        txt.append(f"Empty labels: {empty_labels}\n\n")

        txt.append("Images per split:\n")
        for k, v in per_split.items():
            txt.append(f"  {k}: {v}\n")

        txt.append("\nClasses:\n")
        for c in sorted(set(class_img_counts) | set(class_obj_counts)):
            txt.append(
                f"  {c} {self.class_names.get(c, '')}: "
                f"images={class_img_counts.get(c, 0)}, objects={class_obj_counts.get(c, 0)}\n"
            )

        if covs:
            arr = np.array(covs)
            txt.append("\nLargest bbox coverage:\n")
            txt.append(f"  mean: {arr.mean()*100:.2f}%\n")
            txt.append(f"  median: {np.median(arr)*100:.2f}%\n")
            txt.append(f"  min: {arr.min()*100:.2f}%\n")
            txt.append(f"  max: {arr.max()*100:.2f}%\n")
            txt.append(f"  Very low value candidates (<0.5%): {int((arr < .005).sum())}\n")
            txt.append(f"  Low value candidates (<1%): {int((arr < .01).sum())}\n")
            txt.append(f"  Small objects (1-5%): {int(((arr >= .01) & (arr < .05)).sum())}\n")
            txt.append(f"  Medium objects (5-15%): {int(((arr >= .05) & (arr < .15)).sum())}\n")
            txt.append(f"  Large objects (>15%): {int((arr >= .15).sum())}\n")

        self.analysis.delete("1.0", "end")
        self.analysis.insert("1.0", "".join(txt))
        self.status.set("Dataset analysis complete.")

    def process_threaded(self):
        threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        dataset = Path(self.dataset.get())
        output = Path(self.output.get())

        if not dataset.exists():
            messagebox.showerror("Error", "Input folder does not exist.")
            return

        if output.exists():
            if not messagebox.askyesno("Output exists", "Ο φάκελος εξόδου υπάρχει. Να διαγραφεί;"):
                return
            shutil.rmtree(output)

        output.mkdir(parents=True, exist_ok=True)

        seed = int(self.seed.get() or 42)
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        mult = self.parse_mult() if self.class_aware.get() else {}

        jobs = []
        for split, img_dir, _ in find_splits(dataset):
            for img in images_in(img_dir):
                label = label_for_image(img)
                classes = read_classes(label)
                extra = max([mult.get(c, 1) for c in classes] + [1]) if classes else 1
                n = max(1, int(self.factor.get())) * extra
                jobs.append((split, img, label, classes, n))

        total = sum(j[-1] for j in jobs)
        if total == 0:
            messagebox.showerror("Error", "No jobs.")
            return

        log_fields = [
            "original_image", "new_image", "original_label", "new_label", "split",
            "strength_mode", "selected_strength", "max_label_coverage", "total_label_coverage",
            "object_count", "preset", "strength_scale", "exposure", "gamma", "contrast",
            "saturation", "weather", "noise_sigma", "vignette", "blur", "chromatic_shift",
            "sharpness", "jpeg_quality", "quality_status", "brightness_mean",
            "sharpness_metric", "classes", "attempt"
        ]

        rejected = 0
        done = 0

        with (output / "augmentation_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()

            for split, img, label, classes, n in jobs:
                try:
                    rel_img = img.relative_to(dataset)
                except Exception:
                    rel_img = Path(img.name)

                try:
                    rel_label = label.relative_to(dataset)
                except Exception:
                    rel_label = rel_img.with_suffix(".txt")

                if self.keep_original.get():
                    dest = output / rel_img
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, dest)

                    dl = output / rel_label
                    dl.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label, dl) if label.exists() else dl.write_text("", encoding="utf-8")

                source = Image.open(img).convert("RGB")
                max_cov, total_cov, obj_count = label_coverage_stats(label)

                strength_mode = self.strength.get()
                selected_strength = auto_strength_from_coverage(max_cov, obj_count) if strength_mode == "Auto" else strength_mode

                for i in range(1, n + 1):
                    ok = False
                    params = {}
                    q_status = "not_run"
                    mean = 0.0
                    sharp = 0.0
                    aug = source

                    for attempt in range(1, 6):
                        if strength_mode == "Auto":
                            preset = weighted_choice_for_strength(py_rng, selected_strength)
                        else:
                            preset = weighted_choice(py_rng)

                        aug, params = augment(source, preset, selected_strength, py_rng, np_rng)

                        ok, q_status, mean, sharp = quality_ok(
                            aug,
                            self.quality_filter.get(),
                            float(self.min_b.get()),
                            float(self.max_b.get()),
                            float(self.min_sharp.get())
                        )

                        if ok:
                            break

                    done += 1

                    if not ok:
                        rejected += 1
                        self.progress.set(done / total * 100)
                        continue

                    suffix = f"_s2r_{i:02d}"
                    rel_new = rel_img.with_name(f"{rel_img.stem}{suffix}.jpg")
                    dest_img = output / rel_new
                    dest_img.parent.mkdir(parents=True, exist_ok=True)

                    aug.save(dest_img, quality=int(params.get("jpeg_quality", 85)), optimize=True)

                    rel_new_label = rel_label.with_name(f"{rel_label.stem}{suffix}.txt")
                    dest_label = output / rel_new_label
                    dest_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label, dest_label) if label.exists() else dest_label.write_text("", encoding="utf-8")

                    row = {
                        "original_image": str(rel_img),
                        "new_image": str(rel_new),
                        "original_label": str(rel_label),
                        "new_label": str(rel_new_label),
                        "split": split,
                        "strength_mode": strength_mode,
                        "selected_strength": selected_strength,
                        "max_label_coverage": f"{max_cov:.6f}",
                        "total_label_coverage": f"{total_cov:.6f}",
                        "object_count": obj_count,
                        "quality_status": q_status,
                        "brightness_mean": f"{mean:.2f}",
                        "sharpness_metric": f"{sharp:.2f}",
                        "classes": "|".join(map(str, classes)),
                        "attempt": attempt,
                    }
                    row.update(params)
                    writer.writerow(row)

                    self.progress.set(done / total * 100)
                    self.status.set(f"Processing {done}/{total}: {img.name}")

        self.progress.set(100)
        self.status.set(f"Done. Rejected: {rejected}. Output: {output}")
        messagebox.showinfo("Done", f"Ολοκληρώθηκε.\nOutput: {output}\nRejected: {rejected}\nLog: augmentation_log.csv")


if __name__ == "__main__":
    App().mainloop()
