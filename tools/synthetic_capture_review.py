#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WarThunder Dataset Capture Assistant v1.1 - Fast Capture Mode

Recommended:
pip install pillow mss pynput

Run:
python warthunder_dataset_capture_assistant_v1_1_fast.py
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import mss
    from PIL import Image, ImageTk, ImageOps
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n\n"
        "pip install pillow mss pynput\n\n"
        f"Original error: {exc}"
    )

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR
APP_TITLE = "WarThunder Dataset Capture Assistant v1.1 - Fast Capture Mode"
CONFIG_FILE = DATASET_DIR / "wt_capture_config_v1_1.json"

CLASSES = [
    "CIV_VEHICLE",
    "MIL_ARM_VEHICLE",
    "MIL_ARTILLERY",
    "MIL_MBT",
    "MIL_MISS_LAUNCHER",
    "MIL_VEHICLE",
]

ENVIRONMENTS = ["desert", "forest", "urban", "winter", "open_field", "road", "mixed"]
CAPTURE_TYPES = [
    "random_freecam", "drone_view", "ground_view", "long_range", "medium_range",
    "close_range", "partial_occlusion", "side_view", "front_view", "rear_view"
]
DISTANCE_BANDS = ["auto_unspecified", "30_80m_close", "80_150m_medium", "150_300m_recon", "300m_plus_far"]
HEIGHT_BANDS = ["auto_unspecified", "ground", "low_30_60m", "high_100_200m", "very_high_200m_plus"]
OCCLUSIONS = ["none", "light", "medium", "heavy"]


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass
class CaptureRecord:
    filename_raw: str
    filename_640: str
    class_name: str
    vehicle: str
    environment: str
    map_name: str
    capture_type: str
    distance_band: str
    height_band: str
    occlusion: str
    notes: str
    source: str
    raw_width: int
    raw_height: int
    timestamp: str


class RegionSelector:
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None
        self.region: Optional[CaptureRegion] = None

        self.root = tk.Toplevel(parent)
        self.root.title("Select Capture Region")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.25)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black")

        self.canvas = tk.Canvas(self.root, cursor="cross", bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Escape>", self.cancel)

        self.canvas.create_text(
            40, 40, anchor="nw",
            text="Drag με το mouse γύρω από το παράθυρο/περιοχή gameplay του War Thunder.\nΑφήνεις το mouse για αποθήκευση region. ESC για ακύρωση.",
            fill="white", font=("Segoe UI", 18, "bold")
        )

    def on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=3)

    def on_drag(self, event):
        if self.rect_id:
            x0 = self.start_x - self.root.winfo_rootx()
            y0 = self.start_y - self.root.winfo_rooty()
            self.canvas.coords(self.rect_id, x0, y0, event.x, event.y)

    def on_release(self, event):
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x_root, event.y_root
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        if width < 100 or height < 100:
            messagebox.showwarning("Region too small", "Το selected region είναι πολύ μικρό.")
        else:
            self.region = CaptureRegion(left=left, top=top, width=width, height=height)
        self.root.destroy()

    def cancel(self, event=None):
        self.region = None
        self.root.destroy()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x760")

        self.output_dir = tk.StringVar(value=str(DATASET_DIR / "wt_dataset_capture"))
        self.source = tk.StringVar(value="WT")
        self.class_name = tk.StringVar(value="MIL_MBT")
        self.vehicle = tk.StringVar(value="Leopard2A6")
        self.environment = tk.StringVar(value="desert")
        self.map_name = tk.StringVar(value="Sinai")
        self.capture_type = tk.StringVar(value="random_freecam")
        self.distance_band = tk.StringVar(value="auto_unspecified")
        self.height_band = tk.StringVar(value="auto_unspecified")
        self.occlusion = tk.StringVar(value="none")
        self.notes = tk.StringVar(value="")
        self.jpg_quality = tk.IntVar(value=95)

        self.region: Optional[CaptureRegion] = None
        self.session_count = 0
        self.rejected_count = 0
        self.last_capture: Optional[CaptureRecord] = None
        self.last_raw_path: Optional[Path] = None
        self.last_640_path: Optional[Path] = None
        self.last_preview_img = None
        self.hotkey_listener = None
        self.hotkeys_active = tk.BooleanVar(value=True)

        self.load_config()
        self.build_ui()
        self.update_counts()
        self.update_status()
        self.start_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.tab_setup = ttk.Frame(notebook)
        self.tab_fast = ttk.Frame(notebook)
        self.tab_metadata = ttk.Frame(notebook)
        notebook.add(self.tab_setup, text="1. Session Setup")
        notebook.add(self.tab_fast, text="2. Fast Capture")
        notebook.add(self.tab_metadata, text="3. Metadata / Export")
        self.build_setup_tab()
        self.build_fast_tab()
        self.build_metadata_tab()
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").pack(fill=tk.X, side=tk.BOTTOM)

    def build_setup_tab(self):
        f = ttk.Frame(self.tab_setup, padding=12)
        f.pack(fill=tk.BOTH, expand=True)
        fields = [
            ("Output folder", self.output_dir, "entry"),
            ("Class", self.class_name, CLASSES),
            ("Vehicle name", self.vehicle, "entry"),
            ("Environment", self.environment, ENVIRONMENTS),
            ("Map name", self.map_name, "entry"),
            ("Default capture type", self.capture_type, CAPTURE_TYPES),
            ("Default distance band", self.distance_band, DISTANCE_BANDS),
            ("Default height band", self.height_band, HEIGHT_BANDS),
            ("Default occlusion", self.occlusion, OCCLUSIONS),
            ("Notes", self.notes, "entry"),
        ]
        row = 0
        for label, var, kind in fields:
            ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=5)
            if label == "Output folder":
                ttk.Entry(f, textvariable=var, width=80).grid(row=row, column=1, sticky="we", pady=5)
                ttk.Button(f, text="Browse", command=self.browse_output).grid(row=row, column=2, padx=5)
            elif kind == "entry":
                ttk.Entry(f, textvariable=var, width=80 if label == "Notes" else 35).grid(row=row, column=1, sticky="we" if label == "Notes" else "w", pady=5)
            else:
                ttk.Combobox(f, textvariable=var, values=kind, state="readonly", width=32).grid(row=row, column=1, sticky="w", pady=5)
            row += 1
        ttk.Label(f, text="JPG quality").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Spinbox(f, from_=80, to=100, textvariable=self.jpg_quality, width=8).grid(row=row, column=1, sticky="w", pady=5)
        row += 1
        region_box = ttk.LabelFrame(f, text="Capture Region", padding=10)
        region_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=18)
        self.region_label = ttk.Label(region_box, text="Region: not selected")
        self.region_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=5)
        ttk.Button(region_box, text="Select Game Region", command=self.select_region).grid(row=1, column=0, padx=4)
        ttk.Button(region_box, text="Preview Region", command=self.preview_region).grid(row=1, column=1, padx=4)
        ttk.Button(region_box, text="Save Config", command=self.save_config).grid(row=1, column=2, padx=4)
        ttk.Button(region_box, text="Open Output Folder", command=self.open_output_folder).grid(row=1, column=3, padx=4)
        row += 1
        hotkey_box = ttk.LabelFrame(f, text="Hotkeys", padding=10)
        hotkey_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=5)
        ttk.Checkbutton(hotkey_box, text="Enable global hotkeys", variable=self.hotkeys_active).grid(row=0, column=0, sticky="w")
        ttk.Label(hotkey_box, text="F8 = Capture & Save | F9 = Reject last").grid(row=1, column=0, sticky="w", pady=5)
        f.columnconfigure(1, weight=1)
        self.update_region_label()

    def build_fast_tab(self):
        f = ttk.Frame(self.tab_fast, padding=10)
        f.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(f)
        top.pack(fill=tk.X)
        self.session_count_var = tk.StringVar(value="Captured this session: 0")
        self.folder_count_var = tk.StringVar(value="dataset_640 folder total: 0")
        self.rejected_count_var = tk.StringVar(value="Rejected this session: 0")
        ttk.Label(top, textvariable=self.session_count_var, font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT, padx=8)
        ttk.Label(top, textvariable=self.folder_count_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=18)
        ttk.Label(top, textvariable=self.rejected_count_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=18)
        buttons = ttk.Frame(f)
        buttons.pack(fill=tk.X, pady=10)
        ttk.Button(buttons, text="Capture Now (F8)", command=self.capture_now).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons, text="Reject Last (F9)", command=self.reject_last).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        meta = ttk.LabelFrame(f, text="Current metadata for next captures", padding=8)
        meta.pack(fill=tk.X, pady=8)
        self.fast_meta_var = tk.StringVar()
        ttk.Label(meta, textvariable=self.fast_meta_var, font=("Segoe UI", 10)).pack(anchor="w")
        ttk.Label(meta, text="Tip: Κράτα capture_type=random_freecam για μέγιστη ταχύτητα και άλλαζε μόνο class/vehicle/environment όταν αλλάζει session.").pack(anchor="w", pady=4)
        content = ttk.Frame(f)
        content.pack(fill=tk.BOTH, expand=True)
        preview_frame = ttk.LabelFrame(content, text="Last 640x640 Capture Preview", padding=8)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(preview_frame, text="No capture yet", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        info_frame = ttk.LabelFrame(content, text="Last Capture Info", padding=8)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
        self.last_info = tk.Text(info_frame, width=48, height=26, wrap="word")
        self.last_info.pack(fill=tk.BOTH, expand=True)

    def build_metadata_tab(self):
        f = ttk.Frame(self.tab_metadata, padding=10)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text="metadata.csv γράφεται στο metadata/. Κάθε F8 προσθέτει μία γραμμή. Το F9 μετακινεί RAW και 640 σε rejected/.").pack(anchor="w", pady=5)
        ttk.Button(f, text="Open metadata.csv", command=self.open_metadata).pack(anchor="w", pady=5)
        ttk.Button(f, text="Open session_log.csv", command=self.open_session_log).pack(anchor="w", pady=5)
        self.stats_text = tk.Text(f, height=24, wrap="word")
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=8)
        self.refresh_stats()

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=str(DATASET_DIR))
        if folder:
            self.output_dir.set(folder)
            self.save_config()
            self.update_counts()

    def output_path(self) -> Path:
        return Path(self.output_dir.get()).resolve()

    def ensure_dirs(self):
        base = self.output_path()
        for sub in ["raw", "dataset_640", "metadata", "rejected/raw", "rejected/dataset_640"]:
            (base / sub).mkdir(parents=True, exist_ok=True)

    def save_config(self):
        data = {"output_dir": self.output_dir.get(), "source": self.source.get(), "class_name": self.class_name.get(), "vehicle": self.vehicle.get(), "environment": self.environment.get(), "map_name": self.map_name.get(), "capture_type": self.capture_type.get(), "distance_band": self.distance_band.get(), "height_band": self.height_band.get(), "occlusion": self.occlusion.get(), "notes": self.notes.get(), "jpg_quality": self.jpg_quality.get(), "region": asdict(self.region) if self.region else None}
        CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.update_status("Config saved.")

    def load_config(self):
        if not CONFIG_FILE.exists():
            return
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for attr in ["output_dir", "source", "class_name", "vehicle", "environment", "map_name", "capture_type", "distance_band", "height_band", "occlusion", "notes"]:
                getattr(self, attr).set(data.get(attr, getattr(self, attr).get()))
            self.jpg_quality.set(int(data.get("jpg_quality", self.jpg_quality.get())))
            if data.get("region"):
                self.region = CaptureRegion(**data["region"])
        except Exception:
            pass

    def open_output_folder(self):
        self.ensure_dirs()
        self.open_path(self.output_path())

    def select_region(self):
        selector = RegionSelector(self.root)
        self.root.wait_window(selector.root)
        if selector.region:
            self.region = selector.region
            self.update_region_label()
            self.save_config()

    def update_region_label(self):
        if not hasattr(self, "region_label"):
            return
        if self.region:
            self.region_label.config(text=f"Region: left={self.region.left}, top={self.region.top}, width={self.region.width}, height={self.region.height}")
        else:
            self.region_label.config(text="Region: not selected")

    def preview_region(self):
        try:
            img = self.capture_region_image()
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))
            return
        win = tk.Toplevel(self.root)
        win.title("Region Preview")
        win.geometry("900x650")
        img_thumb = img.copy()
        img_thumb.thumbnail((880, 600))
        tk_img = ImageTk.PhotoImage(img_thumb)
        label = ttk.Label(win, image=tk_img)
        label.image = tk_img
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def capture_region_image(self) -> Image.Image:
        if not self.region:
            raise RuntimeError("Δεν έχεις επιλέξει Capture Region. Πάτα πρώτα Select Game Region.")
        mon = {"left": self.region.left, "top": self.region.top, "width": self.region.width, "height": self.region.height}
        with mss.mss() as sct:
            shot = sct.grab(mon)
            return Image.frombytes("RGB", shot.size, shot.rgb)

    @staticmethod
    def letterbox_square(img: Image.Image, size: int = 640) -> Image.Image:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        scale = min(size / w, size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
        return canvas

    @staticmethod
    def safe_name(text: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        out = "".join(c if c in allowed else "_" for c in text.strip())
        return out.strip("_") or "unknown"

    def next_filename_stem(self) -> str:
        self.ensure_dirs()
        base = f"{self.source.get()}_{self.class_name.get()}_{self.safe_name(self.vehicle.get())}_{self.environment.get()}_{self.safe_name(self.map_name.get())}_{self.safe_name(self.capture_type.get())}"
        existing = sorted((self.output_path() / "raw").glob(base + "_*.jpg"))
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem.split("_")[-1]))
            except Exception:
                pass
        return f"{base}_{max(nums, default=0) + 1:04d}"

    def capture_now(self):
        try:
            self.ensure_dirs()
            img = self.capture_region_image()
            raw_w, raw_h = img.size
            stem = self.next_filename_stem()
            raw_path = self.output_path() / "raw" / f"{stem}.jpg"
            out640_path = self.output_path() / "dataset_640" / f"{stem}.jpg"
            q = int(self.jpg_quality.get())
            img.save(raw_path, "JPEG", quality=q, optimize=True)
            self.letterbox_square(img, 640).save(out640_path, "JPEG", quality=q, optimize=True)
            record = CaptureRecord(str(raw_path.relative_to(self.output_path())), str(out640_path.relative_to(self.output_path())), self.class_name.get(), self.vehicle.get(), self.environment.get(), self.map_name.get(), self.capture_type.get(), self.distance_band.get(), self.height_band.get(), self.occlusion.get(), self.notes.get(), self.source.get(), raw_w, raw_h, datetime.now().isoformat(timespec="seconds"))
            self.append_metadata(record)
            self.append_session_log("capture", record.filename_raw)
            self.session_count += 1
            self.last_capture = record
            self.last_raw_path = raw_path
            self.last_640_path = out640_path
            self.show_preview(out640_path)
            self.update_last_info()
            self.update_counts()
            self.refresh_stats()
            self.update_status(f"Captured: {out640_path.name}")
        except Exception as exc:
            messagebox.showerror("Capture failed", str(exc))

    def append_metadata(self, record: CaptureRecord):
        path = self.output_path() / "metadata" / "metadata.csv"
        file_exists = path.exists()
        fields = list(asdict(record).keys())
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(record))

    def append_session_log(self, action: str, item: str):
        path = self.output_path() / "metadata" / "session_log.csv"
        file_exists = path.exists()
        fields = ["timestamp", "action", "item", "class_name", "vehicle", "environment", "map_name", "capture_type"]
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "action": action, "item": item, "class_name": self.class_name.get(), "vehicle": self.vehicle.get(), "environment": self.environment.get(), "map_name": self.map_name.get(), "capture_type": self.capture_type.get()}
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def reject_last(self):
        if not self.last_capture or not self.last_raw_path or not self.last_640_path:
            self.update_status("No last capture to reject.")
            return
        try:
            self.ensure_dirs()
            rejected_raw = self.output_path() / "rejected" / "raw" / self.last_raw_path.name
            rejected_640 = self.output_path() / "rejected" / "dataset_640" / self.last_640_path.name
            if self.last_raw_path.exists():
                shutil.move(str(self.last_raw_path), str(rejected_raw))
            if self.last_640_path.exists():
                shutil.move(str(self.last_640_path), str(rejected_640))
            self.session_count = max(0, self.session_count - 1)
            self.rejected_count += 1
            self.append_session_log("reject", self.last_capture.filename_raw)
            self.last_capture = None
            self.last_raw_path = None
            self.last_640_path = None
            self.preview_label.config(image="", text="Last capture rejected")
            self.last_info.delete("1.0", tk.END)
            self.last_info.insert(tk.END, "Last capture rejected.\n")
            self.update_counts()
            self.refresh_stats()
            self.update_status("Rejected last capture.")
        except Exception as exc:
            messagebox.showerror("Reject failed", str(exc))

    def count_dataset_files(self) -> int:
        try:
            return len(list((self.output_path() / "dataset_640").glob("*.jpg")))
        except Exception:
            return 0

    def update_counts(self):
        if hasattr(self, "session_count_var"):
            self.session_count_var.set(f"Captured this session: {self.session_count}")
            self.folder_count_var.set(f"dataset_640 folder total: {self.count_dataset_files()}")
            self.rejected_count_var.set(f"Rejected this session: {self.rejected_count}")
        if hasattr(self, "fast_meta_var"):
            self.fast_meta_var.set(f"class={self.class_name.get()} | vehicle={self.vehicle.get()} | environment={self.environment.get()} | map={self.map_name.get()} | type={self.capture_type.get()}")

    def refresh_stats(self):
        if not hasattr(self, "stats_text"):
            return
        meta = self.output_path() / "metadata" / "metadata.csv"
        lines = ["Fast Capture statistics", f"Captured this GUI session: {self.session_count}", f"Rejected this GUI session: {self.rejected_count}", f"dataset_640 folder total: {self.count_dataset_files()}", ""]
        if meta.exists():
            try:
                rows = list(csv.DictReader(meta.open("r", encoding="utf-8")))
                lines.append(f"metadata.csv total rows: {len(rows)}")
                for field, title in [("class_name", "By class"), ("environment", "By environment"), ("vehicle", "By vehicle")]:
                    counts = {}
                    for r in rows:
                        k = r.get(field, "unknown")
                        counts[k] = counts.get(k, 0) + 1
                    lines += ["", f"{title}:"] + [f"- {k}: {v}" for k, v in sorted(counts.items())]
            except Exception as exc:
                lines.append(f"Could not read metadata.csv: {exc}")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "\n".join(lines))

    def show_preview(self, path: Path):
        img = Image.open(path)
        img.thumbnail((720, 520))
        self.last_preview_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self.last_preview_img, text="")

    def update_last_info(self):
        self.last_info.delete("1.0", tk.END)
        if self.last_capture:
            for k, v in asdict(self.last_capture).items():
                self.last_info.insert(tk.END, f"{k}: {v}\n")

    def open_metadata(self):
        self.open_path(self.output_path() / "metadata" / "metadata.csv")

    def open_session_log(self):
        self.open_path(self.output_path() / "metadata" / "session_log.csv")

    @staticmethod
    def open_path(path: Optional[Path]):
        if not path:
            return
        path = Path(path)
        if not path.exists():
            messagebox.showwarning("Not found", f"Δεν βρέθηκε:\n{path}")
            return
        import os, subprocess, sys
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def start_hotkeys(self):
        if not PYNPUT_AVAILABLE:
            messagebox.showwarning("Hotkeys disabled", "Δεν φορτώθηκε το pynput. Τα κουμπιά μέσα στο GUI δουλεύουν, αλλά όχι τα global hotkeys.\n\nInstall:\npip install pynput")
            return
        def on_press(key):
            if not self.hotkeys_active.get():
                return
            try:
                if key == keyboard.Key.f8:
                    self.root.after(0, self.capture_now)
                elif key == keyboard.Key.f9:
                    self.root.after(0, self.reject_last)
            except Exception:
                pass
        self.hotkey_listener = keyboard.Listener(on_press=on_press)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()

    def update_status(self, text: Optional[str] = None):
        if text is None:
            text = f"Ready | Fast Capture Mode | Region: {'selected' if self.region else 'not selected'}"
        self.status_var.set(text)
        self.update_counts()

    def on_close(self):
        self.save_config()
        try:
            if self.hotkey_listener:
                self.hotkey_listener.stop()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.15)
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
