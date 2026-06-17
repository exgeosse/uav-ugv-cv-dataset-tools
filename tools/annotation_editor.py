#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Class Editor GUI v7.1 - CVAT-style Zoom/Pan + Safe Hotkeys

Based on yolo_class_editor_gui_v6_resize_drag_crosshair.py

Keeps the original dataset structure support:
  dataset/
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images, test/labels
  OR
    images/train, labels/train
    images/valid, labels/valid
    images/test, labels/test

Adds:
  Mouse Wheel = zoom-to-cursor
  Middle Mouse Drag = pan
  F = fit image
  Full-window image rendering
  Improved resize handles
  Space = Save & Next
  A/D = Previous/Next
  Delete = Delete selected box
  1-9 / 0 = set selected box class
  Ctrl+S = Save
  R = delete/reject current image+label after confirmation

Dependencies:
  pip install pillow pyyaml
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import yaml
import shutil
import csv
from datetime import datetime

# =====================================================
# ΡΥΘΜΙΣΕΙΣ
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]

CREATE_BACKUP = True

BOX_COLOR = "red"
SELECTED_BOX_COLOR = "lime"
BOX_WIDTH = 3
HANDLE_SIZE = 10
CROSSHAIR_COLOR = "#d9d9d9"
CROSSHAIR_DASH = (6, 4)

MIN_ZOOM = 0.2
MAX_ZOOM = 16.0
ZOOM_STEP = 1.25

# =====================================================


class YOLOAnnotationEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotation Editor v7.1 - Zoom/Pan + Safe Hotkeys")
        self.root.geometry("1280x900")
        self.root.minsize(900, 650)

        self.dataset_dir = DATASET_DIR
        self.data_yaml = self.find_yaml_file()

        self.class_names = self.load_classes()
        self.all_images = self.find_images()
        self.images = list(self.all_images)
        self.index = 0
        self.active_filter_class_id = None

        self.annotations = []
        self.selected_box_indices = set()

        self.original_image = None
        self.tk_image = None
        self.rendered_image_id = None

        # CVAT-style viewport state
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # mouse interaction state
        self.drawing = False
        self.start_img_x = None
        self.start_img_y = None
        self.action_mode = None  # draw | move | resize | pan
        self.resize_handle = None
        self.drag_last_img_x = None
        self.drag_last_img_y = None
        self.pan_start_canvas = None
        self.pan_start_offset = None
        self.crosshair_items = []

        self.log_file = self.dataset_dir / "annotation_changes_log.csv"

        self.setup_ui()
        self.bind_hotkeys()

        if not self.images:
            messagebox.showerror("Error", "Δεν βρέθηκαν εικόνες στο dataset.")
            self.root.destroy()
            return

        self.load_current_image()

    # =====================================================
    # DATASET / YAML
    # =====================================================

    def find_yaml_file(self):
        for name in ["data.yaml", "data.yml"]:
            path = self.dataset_dir / name
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Δεν βρέθηκε data.yaml ή data.yml στον φάκελο:\n{self.dataset_dir}"
        )

    def load_classes(self):
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", {})

        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}

        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

        raise ValueError("Δεν βρέθηκαν σωστά class names στο data.yaml/data.yml.")

    def find_images(self):
        images = []

        for split in SPLITS:
            image_dir = self.dataset_dir / split / "images"
            if image_dir.exists():
                for img in image_dir.rglob("*"):
                    if img.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append((split, img))

        for split in SPLITS:
            image_dir = self.dataset_dir / "images" / split
            if image_dir.exists():
                for img in image_dir.rglob("*"):
                    if img.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append((split, img))

        return sorted(images, key=lambda x: str(x[1]))

    def get_label_path(self, split, image_path):
        stem = image_path.stem
        candidates = [
            self.dataset_dir / split / "labels" / f"{stem}.txt",
            self.dataset_dir / "labels" / split / f"{stem}.txt",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    # =====================================================
    # UI
    # =====================================================

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        top_bar = ttk.Frame(main)
        top_bar.pack(fill=tk.X, pady=(0, 5))

        self.file_label = ttk.Label(top_bar, text="", font=("Segoe UI", 9))
        self.file_label.pack(side=tk.LEFT, padx=4)

        ttk.Button(top_bar, text="Fit (F)", command=self.fit_image, takefocus=0).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top_bar, text="Save & Next (Space)", command=self.save_and_next, takefocus=0).pack(side=tk.RIGHT, padx=3)
        ttk.Button(top_bar, text="Save (Ctrl+S)", command=lambda: self.save_annotations(show_popup=False), takefocus=0).pack(side=tk.RIGHT, padx=3)

        self.canvas = tk.Canvas(main, bg="black", cursor="cross", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.clear_crosshair)

        # Windows / Linux wheel support
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        # Middle button pan
        self.canvas.bind("<ButtonPress-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)

        info = ttk.Frame(main)
        info.pack(fill=tk.X, pady=5)

        self.class_info_label = ttk.Label(info, text="", font=("Segoe UI", 10, "bold"))
        self.class_info_label.pack(side=tk.LEFT, anchor="w")

        self.progress_label = ttk.Label(info, text="", font=("Segoe UI", 10))
        self.progress_label.pack(side=tk.RIGHT, padx=8)

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=5)

        ttk.Label(controls, text="Νέα class:").pack(side=tk.LEFT)

        self.class_var = tk.StringVar()
        self.class_values = [f"{i}: {name}" for i, name in self.class_names.items()]
        self.class_combo = ttk.Combobox(
            controls,
            textvariable=self.class_var,
            values=self.class_values,
            state="readonly",
            width=38
        )
        self.class_combo.pack(side=tk.LEFT, padx=6)

        ttk.Button(controls, text="Change ALL", command=self.change_all_classes, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Change SELECTED", command=self.change_selected_class, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Delete SELECTED", command=self.delete_selected_box, takefocus=0).pack(side=tk.LEFT, padx=4)

        nav = ttk.Frame(main)
        nav.pack(fill=tk.X, pady=5)

        ttk.Button(nav, text="Previous (A)", command=self.previous_image, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Next (D)", command=self.next_image, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Save & Next (Space)", command=self.save_and_next, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Delete Pair (R)", command=self.delete_current_pair, takefocus=0).pack(side=tk.LEFT, padx=4)

        ttk.Label(nav, text="Go to:").pack(side=tk.LEFT, padx=(20, 4))
        self.goto_var = tk.StringVar()
        self.goto_entry = ttk.Entry(nav, textvariable=self.goto_var, width=8)
        self.goto_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Go", command=self.go_to_image, takefocus=0).pack(side=tk.LEFT, padx=4)

        ttk.Label(nav, text="Batch from:").pack(side=tk.LEFT, padx=(20, 4))
        self.batch_from_var = tk.StringVar()
        self.batch_from_entry = ttk.Entry(nav, textvariable=self.batch_from_var, width=6)
        self.batch_from_entry.pack(side=tk.LEFT, padx=2)

        ttk.Label(nav, text="to:").pack(side=tk.LEFT, padx=2)
        self.batch_to_var = tk.StringVar()
        self.batch_to_entry = ttk.Entry(nav, textvariable=self.batch_to_var, width=6)
        self.batch_to_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(nav, text="Batch Change", command=self.batch_change_range, takefocus=0).pack(side=tk.LEFT, padx=4)

        filter_frame = ttk.LabelFrame(main, text="Filter images by class tag")
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Show images containing at least 1 tag from class:").pack(side=tk.LEFT, padx=5, pady=5)

        self.filter_class_var = tk.StringVar()
        self.filter_class_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_class_var,
            values=self.class_values,
            state="readonly",
            width=38
        )
        self.filter_class_combo.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(filter_frame, text="Apply Class Filter", command=self.apply_class_filter, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(filter_frame, text="Show All Images", command=self.clear_class_filter, takefocus=0).pack(side=tk.LEFT, padx=4)

        export_frame = ttk.LabelFrame(main, text="Copy Current image+txt to selected folder")
        export_frame.pack(fill=tk.X, pady=5)

        ttk.Label(export_frame, text="Target folder:").pack(side=tk.LEFT, padx=5, pady=5)
        self.copy_target_var = tk.StringVar()
        self.copy_target_entry = ttk.Entry(export_frame, textvariable=self.copy_target_var, width=80)
        self.copy_target_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        ttk.Button(export_frame, text="Browse", command=self.choose_copy_target_folder, takefocus=0).pack(side=tk.LEFT, padx=4)
        ttk.Button(export_frame, text="Copy Current jpg+txt", command=self.copy_current_pair_to_folder, takefocus=0).pack(side=tk.LEFT, padx=4)

        help_text = (
            "Mouse Wheel=zoom-to-cursor | Middle Drag=pan | F=fit | "
            "A/D prev-next | Space save&next | R delete pair | Ctrl+S save | "
            "Delete selected box | 1-9/0 set selected class | "
            "Click box select | Ctrl+Click multi-select | Drag empty=new box | Drag box=move | Drag green corner=resize"
        )
        ttk.Label(main, text=help_text).pack(anchor="w", pady=2)

    def bind_hotkeys(self):
        # Safer hotkeys:
        # bind_all + return "break" prevents Space/Enter from activating whichever ttk.Button has focus.
        def handled(fn):
            def wrapper(event=None):
                fn()
                return "break"
            return wrapper

        self.root.bind_all("<a>", handled(self.previous_image))
        self.root.bind_all("<A>", handled(self.previous_image))
        self.root.bind_all("<d>", handled(self.next_image))
        self.root.bind_all("<D>", handled(self.next_image))
        self.root.bind_all("<space>", handled(self.save_and_next))
        self.root.bind_all("<r>", handled(self.delete_current_pair))
        self.root.bind_all("<R>", handled(self.delete_current_pair))
        self.root.bind_all("<f>", handled(self.fit_image))
        self.root.bind_all("<F>", handled(self.fit_image))
        self.root.bind_all("<Delete>", handled(lambda: self.delete_selected_box(show_warning=False)))
        self.root.bind_all("<Control-s>", handled(lambda: self.save_annotations(show_popup=False)))
        self.root.bind_all("<Control-S>", handled(lambda: self.save_annotations(show_popup=False)))

        # 1-9 maps to class 0-8, 0 maps to class 9.
        for key in "123456789":
            class_id = int(key) - 1
            self.root.bind_all(key, lambda e, cls=class_id: (self.set_selected_class_by_id(cls), "break")[1])
        self.root.bind_all("0", lambda e: (self.set_selected_class_by_id(9), "break")[1])

    # =====================================================
    # LOAD / SAVE
    # =====================================================

    def load_current_image(self):
        split, image_path = self.images[self.index]

        self.current_split = split
        self.current_image_path = image_path
        self.current_label_path = self.get_label_path(split, image_path)

        self.original_image = Image.open(image_path).convert("RGB")
        self.annotations = self.read_annotations(self.current_label_path)
        self.selected_box_indices = set()

        if self.class_names:
            first_id = list(self.class_names.keys())[0]
            self.class_var.set(f"{first_id}: {self.class_names[first_id]}")
            if hasattr(self, "filter_class_var") and not self.filter_class_var.get():
                self.filter_class_var.set(f"{first_id}: {self.class_names[first_id]}")

        self.update_info()
        self.root.after(50, self.fit_image)

    def read_annotations(self, label_path):
        annotations = []
        if not label_path.exists():
            return annotations

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                annotations.append({
                    "class_id": int(float(parts[0])),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4])
                })
            except ValueError:
                continue

        return annotations

    def save_annotations(self, show_popup=True):
        label_path = self.current_label_path

        if CREATE_BACKUP and label_path.exists():
            backup_path = label_path.with_suffix(".txt.bak")
            if not backup_path.exists():
                shutil.copy2(label_path, backup_path)

        lines = []
        for ann in self.annotations:
            x = max(0, min(1, ann["x"]))
            y = max(0, min(1, ann["y"]))
            w = max(0, min(1, ann["w"]))
            h = max(0, min(1, ann["h"]))
            lines.append(f'{ann["class_id"]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        self.write_log()

        if show_popup:
            messagebox.showinfo("Saved", "Το annotation αποθηκεύτηκε.")

        self.update_info()

    def write_log(self):
        file_exists = self.log_file.exists()
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "image", "label", "objects"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                str(self.current_image_path),
                str(self.current_label_path),
                len(self.annotations)
            ])

    # =====================================================
    # COORDINATES / VIEWPORT
    # =====================================================

    def canvas_to_image(self, cx, cy):
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    def image_to_canvas(self, ix, iy):
        return ix * self.zoom + self.pan_x, iy * self.zoom + self.pan_y

    def clamp_image_point(self, ix, iy):
        if self.original_image is None:
            return ix, iy
        w, h = self.original_image.size
        return max(0, min(ix, w)), max(0, min(iy, h))

    def fit_image(self):
        if self.original_image is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.original_image.size
        self.fit_zoom = min(cw / iw, ch / ih)
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.fit_zoom))
        self.pan_x = (cw - iw * self.zoom) / 2
        self.pan_y = (ch - ih * self.zoom) / 2
        self.render_image()

    def on_canvas_resize(self, event=None):
        if self.original_image is None:
            return
        if abs(self.zoom - self.fit_zoom) < 0.01:
            self.fit_image()
        else:
            self.render_image()

    def on_mouse_wheel(self, event):
        if self.original_image is None:
            return

        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = ZOOM_STEP
        else:
            factor = 1 / ZOOM_STEP

        old_zoom = self.zoom
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)
        self.zoom = new_zoom
        self.pan_x = event.x - ix * self.zoom
        self.pan_y = event.y - iy * self.zoom
        self.render_image()
        self.draw_crosshair(event.x, event.y)

    def on_middle_down(self, event):
        self.action_mode = "pan"
        self.pan_start_canvas = (event.x, event.y)
        self.pan_start_offset = (self.pan_x, self.pan_y)
        self.canvas.config(cursor="fleur")

    def on_middle_drag(self, event):
        if self.action_mode != "pan" or not self.pan_start_canvas:
            return
        sx, sy = self.pan_start_canvas
        ox, oy = self.pan_start_offset
        self.pan_x = ox + (event.x - sx)
        self.pan_y = oy + (event.y - sy)
        self.render_image()

    def on_middle_up(self, event):
        self.action_mode = None
        self.pan_start_canvas = None
        self.pan_start_offset = None
        self.canvas.config(cursor="cross")

    # =====================================================
    # RENDER
    # =====================================================

    def render_image(self):
        if self.original_image is None:
            return

        self.canvas.delete("all")

        iw, ih = self.original_image.size
        disp_w = max(1, int(iw * self.zoom))
        disp_h = max(1, int(ih * self.zoom))

        img = self.original_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_image)
        self.draw_boxes()
        self.update_info()

    def draw_boxes(self):
        if self.original_image is None:
            return

        for idx, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = self.annotation_to_image_box(ann)
            cx1, cy1 = self.image_to_canvas(x1, y1)
            cx2, cy2 = self.image_to_canvas(x2, y2)

            color = SELECTED_BOX_COLOR if idx in self.selected_box_indices else BOX_COLOR

            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=BOX_WIDTH, tags="bbox")

            class_name = self.class_names.get(ann["class_id"], str(ann["class_id"]))
            self.canvas.create_text(
                cx1, max(10, cy1 - 10),
                text=class_name,
                fill=color,
                anchor=tk.W,
                font=("Segoe UI", 10, "bold"),
                tags="bbox"
            )

            if idx in self.selected_box_indices:
                hs = HANDLE_SIZE / 2
                for hx, hy in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]:
                    self.canvas.create_rectangle(
                        hx - hs, hy - hs, hx + hs, hy + hs,
                        fill=color,
                        outline="black",
                        tags="bbox"
                    )

    def annotation_to_image_box(self, ann):
        iw, ih = self.original_image.size
        x_center = ann["x"] * iw
        y_center = ann["y"] * ih
        box_w = ann["w"] * iw
        box_h = ann["h"] * ih
        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2
        return x1, y1, x2, y2

    def update_annotation_from_image_box(self, idx, x1, y1, x2, y2):
        iw, ih = self.original_image.size
        x1, y1 = self.clamp_image_point(x1, y1)
        x2, y2 = self.clamp_image_point(x2, y2)

        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)

        if right - left < 2 or bottom - top < 2:
            return

        self.annotations[idx]["x"] = ((left + right) / 2) / iw
        self.annotations[idx]["y"] = ((top + bottom) / 2) / ih
        self.annotations[idx]["w"] = (right - left) / iw
        self.annotations[idx]["h"] = (bottom - top) / ih

    # =====================================================
    # HIT TEST / CROSSHAIR
    # =====================================================

    def find_box_at_position(self, canvas_x, canvas_y):
        ix, iy = self.canvas_to_image(canvas_x, canvas_y)
        for idx in range(len(self.annotations) - 1, -1, -1):
            x1, y1, x2, y2 = self.annotation_to_image_box(self.annotations[idx])
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return idx
        return None

    def get_handle_at_position(self, canvas_x, canvas_y):
        if len(self.selected_box_indices) != 1:
            return None, None

        idx = next(iter(self.selected_box_indices))
        x1, y1, x2, y2 = self.annotation_to_image_box(self.annotations[idx])

        handles = {
            "nw": self.image_to_canvas(x1, y1),
            "ne": self.image_to_canvas(x2, y1),
            "sw": self.image_to_canvas(x1, y2),
            "se": self.image_to_canvas(x2, y2),
        }

        for handle, (hx, hy) in handles.items():
            if abs(canvas_x - hx) <= HANDLE_SIZE and abs(canvas_y - hy) <= HANDLE_SIZE:
                return idx, handle

        return None, None

    def draw_crosshair(self, x, y):
        self.clear_crosshair()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.crosshair_items = [
            self.canvas.create_line(0, y, cw, y, fill=CROSSHAIR_COLOR, dash=CROSSHAIR_DASH, width=1, tags="crosshair"),
            self.canvas.create_line(x, 0, x, ch, fill=CROSSHAIR_COLOR, dash=CROSSHAIR_DASH, width=1, tags="crosshair"),
        ]

    def clear_crosshair(self, event=None):
        self.canvas.delete("crosshair")
        self.crosshair_items = []

    # =====================================================
    # MOUSE ACTIONS
    # =====================================================

    def on_mouse_move(self, event):
        if self.original_image is None:
            return

        idx, handle = self.get_handle_at_position(event.x, event.y)

        if handle in ("nw", "se"):
            self.canvas.config(cursor="size_nw_se")
        elif handle in ("ne", "sw"):
            self.canvas.config(cursor="size_ne_sw")
        elif self.find_box_at_position(event.x, event.y) is not None:
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="cross")

        self.draw_crosshair(event.x, event.y)

    def on_mouse_down(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        ix, iy = self.clamp_image_point(ix, iy)
        ctrl_pressed = (event.state & 0x0004) != 0

        handle_idx, handle = self.get_handle_at_position(event.x, event.y)
        if handle_idx is not None:
            self.action_mode = "resize"
            self.resize_handle = handle
            return

        clicked_idx = self.find_box_at_position(event.x, event.y)

        if clicked_idx is not None:
            if ctrl_pressed:
                if clicked_idx in self.selected_box_indices:
                    self.selected_box_indices.remove(clicked_idx)
                    self.action_mode = None
                else:
                    self.selected_box_indices.add(clicked_idx)
                    self.action_mode = "move"
            else:
                self.selected_box_indices = {clicked_idx}
                self.action_mode = "move"

            self.drag_last_img_x = ix
            self.drag_last_img_y = iy
            self.render_image()
            self.draw_crosshair(event.x, event.y)
            return

        if not ctrl_pressed:
            self.selected_box_indices.clear()

        self.action_mode = "draw"
        self.drawing = True
        self.start_img_x = ix
        self.start_img_y = iy
        self.render_image()
        self.draw_crosshair(event.x, event.y)

    def on_mouse_drag(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        ix, iy = self.clamp_image_point(ix, iy)

        if self.action_mode == "move" and self.selected_box_indices:
            dx = ix - self.drag_last_img_x
            dy = iy - self.drag_last_img_y
            iw, ih = self.original_image.size

            for idx in self.selected_box_indices:
                ann = self.annotations[idx]
                box_w = ann["w"] * iw
                box_h = ann["h"] * ih

                new_x_px = ann["x"] * iw + dx
                new_y_px = ann["y"] * ih + dy

                new_x_px = max(box_w / 2, min(iw - box_w / 2, new_x_px))
                new_y_px = max(box_h / 2, min(ih - box_h / 2, new_y_px))

                ann["x"] = new_x_px / iw
                ann["y"] = new_y_px / ih

            self.drag_last_img_x = ix
            self.drag_last_img_y = iy
            self.render_image()
            self.draw_crosshair(event.x, event.y)
            return

        if self.action_mode == "resize" and len(self.selected_box_indices) == 1:
            idx = next(iter(self.selected_box_indices))
            x1, y1, x2, y2 = self.annotation_to_image_box(self.annotations[idx])

            if self.resize_handle == "nw":
                x1, y1 = ix, iy
            elif self.resize_handle == "ne":
                x2, y1 = ix, iy
            elif self.resize_handle == "sw":
                x1, y2 = ix, iy
            elif self.resize_handle == "se":
                x2, y2 = ix, iy

            self.update_annotation_from_image_box(idx, x1, y1, x2, y2)
            self.render_image()
            self.draw_crosshair(event.x, event.y)
            return

        if self.action_mode == "draw" and self.drawing:
            self.render_image()
            cx1, cy1 = self.image_to_canvas(self.start_img_x, self.start_img_y)
            cx2, cy2 = self.image_to_canvas(ix, iy)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="yellow", width=2, tags="temp")
            self.draw_crosshair(event.x, event.y)

    def on_mouse_up(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        ix, iy = self.clamp_image_point(ix, iy)

        if self.action_mode in ("move", "resize"):
            self.action_mode = None
            self.resize_handle = None
            self.drag_last_img_x = None
            self.drag_last_img_y = None
            self.render_image()
            self.draw_crosshair(event.x, event.y)
            return

        if not self.drawing:
            self.action_mode = None
            return

        self.drawing = False
        self.action_mode = None

        x1 = min(self.start_img_x, ix)
        y1 = min(self.start_img_y, iy)
        x2 = max(self.start_img_x, ix)
        y2 = max(self.start_img_y, iy)

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.render_image()
            self.draw_crosshair(event.x, event.y)
            return

        iw, ih = self.original_image.size
        class_id = int(self.class_var.get().split(":")[0])

        self.annotations.append({
            "class_id": class_id,
            "x": ((x1 + x2) / 2) / iw,
            "y": ((y1 + y2) / 2) / ih,
            "w": (x2 - x1) / iw,
            "h": (y2 - y1) / ih
        })

        self.selected_box_indices = {len(self.annotations) - 1}
        self.render_image()
        self.draw_crosshair(event.x, event.y)

    # =====================================================
    # CLASS / BOX ACTIONS
    # =====================================================

    def change_all_classes(self):
        class_id = int(self.class_var.get().split(":")[0])
        for ann in self.annotations:
            ann["class_id"] = class_id
        self.render_image()

    def change_selected_class(self):
        if not self.selected_box_indices:
            messagebox.showwarning("Warning", "Δεν έχει επιλεγεί κανένα box.")
            return

        class_id = int(self.class_var.get().split(":")[0])
        for idx in self.selected_box_indices:
            self.annotations[idx]["class_id"] = class_id

        self.render_image()

    def set_selected_class_by_id(self, class_id):
        if class_id not in self.class_names:
            return

        self.class_var.set(f"{class_id}: {self.class_names[class_id]}")

        if self.selected_box_indices:
            for idx in self.selected_box_indices:
                self.annotations[idx]["class_id"] = class_id
            self.render_image()

    def delete_selected_box(self, show_warning=True):
        if not self.selected_box_indices:
            if show_warning:
                messagebox.showwarning("Warning", "Δεν έχει επιλεγεί κανένα box.")
            return

        for idx in sorted(self.selected_box_indices, reverse=True):
            del self.annotations[idx]

        self.selected_box_indices.clear()
        self.render_image()

    # =====================================================
    # NAVIGATION / FILTERS
    # =====================================================

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_current_image()
        else:
            messagebox.showinfo("End", "Τελευταία εικόνα.")

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_current_image()

    def save_and_next(self):
        self.save_annotations(show_popup=False)
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_current_image()
        else:
            messagebox.showinfo("End", "Τελευταία εικόνα.")

    def go_to_image(self):
        try:
            target = int(self.goto_var.get())
        except ValueError:
            messagebox.showwarning("Warning", "Βάλε αριθμό εικόνας, π.χ. 456.")
            return

        if target < 1 or target > len(self.images):
            messagebox.showwarning("Warning", f"Ο αριθμός πρέπει να είναι από 1 έως {len(self.images)}.")
            return

        self.index = target - 1
        self.load_current_image()

    def read_label_class_ids(self, label_path):
        class_ids = []
        if not label_path.exists():
            return class_ids

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_ids.append(int(float(parts[0])))
            except ValueError:
                continue

        return class_ids

    def image_contains_class(self, split, image_path, class_id):
        label_path = self.get_label_path(split, image_path)
        return class_id in self.read_label_class_ids(label_path)

    def apply_class_filter(self):
        if not self.filter_class_var.get():
            messagebox.showwarning("Warning", "Επίλεξε πρώτα class για φιλτράρισμα.")
            return

        class_id = int(self.filter_class_var.get().split(":")[0])
        filtered = [
            (split, image_path)
            for split, image_path in self.all_images
            if self.image_contains_class(split, image_path, class_id)
        ]

        if not filtered:
            messagebox.showinfo(
                "No Results",
                f"Δεν βρέθηκαν εικόνες που να περιέχουν τουλάχιστον ένα tag από την class:\n"
                f"{class_id}: {self.class_names.get(class_id, '')}"
            )
            return

        self.images = filtered
        self.index = 0
        self.active_filter_class_id = class_id
        self.load_current_image()

    def clear_class_filter(self):
        self.all_images = self.find_images()
        self.images = list(self.all_images)
        self.index = 0
        self.active_filter_class_id = None

        if not self.images:
            messagebox.showerror("Error", "Δεν βρέθηκαν εικόνες στο dataset.")
            self.root.destroy()
            return

        self.load_current_image()

    def update_info(self):
        if self.original_image is None:
            return

        classes = sorted(set(a["class_id"] for a in self.annotations))
        if classes:
            text = ", ".join(f"{cid}: {self.class_names.get(cid, 'unknown')}" for cid in classes)
        else:
            text = "No annotations"

        self.file_label.config(text=f"Image: {self.current_image_path}")
        self.class_info_label.config(text=f"Classes: {text}")

        if self.active_filter_class_id is None:
            filter_text = "All images"
        else:
            filter_text = f"Filtered: {self.active_filter_class_id}: {self.class_names.get(self.active_filter_class_id, '')}"

        self.progress_label.config(
            text=f"{self.index + 1}/{len(self.images)} | Split: {self.current_split} | {filter_text} | Zoom: {self.zoom:.2f}x"
        )

    # =====================================================
    # DELETE / EXPORT / BATCH
    # =====================================================

    def delete_current_pair(self):
        confirm = messagebox.askyesno(
            "Confirm Delete",
            "Θέλεις σίγουρα να διαγράψεις ΟΡΙΣΤΙΚΑ την τρέχουσα εικόνα και το αντίστοιχο txt label;"
        )

        if not confirm:
            return

        image_path = self.current_image_path
        label_path = self.current_label_path

        try:
            if image_path.exists():
                image_path.unlink()

            if label_path.exists():
                label_path.unlink()

            current_pair = (self.current_split, image_path)
            if current_pair in self.all_images:
                self.all_images.remove(current_pair)

            del self.images[self.index]

            if not self.images:
                messagebox.showinfo("Done", "Δεν υπάρχουν άλλες εικόνες.")
                self.root.destroy()
                return

            if self.index >= len(self.images):
                self.index = len(self.images) - 1

            self.load_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"Δεν μπόρεσε να γίνει διαγραφή:\n{e}")

    def batch_change_range(self):
        try:
            start = int(self.batch_from_var.get())
            end = int(self.batch_to_var.get())
        except ValueError:
            messagebox.showwarning("Warning", "Βάλε σωστούς αριθμούς, π.χ. 270 έως 350.")
            return

        if start < 1 or end < 1 or start > len(self.images) or end > len(self.images):
            messagebox.showwarning("Warning", f"Οι αριθμοί πρέπει να είναι από 1 έως {len(self.images)}.")
            return

        if start > end:
            start, end = end, start

        class_id = int(self.class_var.get().split(":")[0])
        class_name = self.class_names.get(class_id, str(class_id))

        confirm = messagebox.askyesno(
            "Confirm Batch Change",
            f"Θέλεις να αλλάξεις ΟΛΑ τα labels στις εικόνες:\n\n"
            f"{start} έως {end}\n\n"
            f"σε class:\n\n"
            f"{class_id}: {class_name};"
        )

        if not confirm:
            return

        changed_files = 0
        changed_objects = 0
        missing_labels = 0

        for idx in range(start - 1, end):
            split, image_path = self.images[idx]
            label_path = self.get_label_path(split, image_path)

            if not label_path.exists():
                missing_labels += 1
                continue

            if CREATE_BACKUP:
                backup_path = label_path.with_suffix(".txt.bak")
                if not backup_path.exists():
                    shutil.copy2(label_path, backup_path)

            lines = label_path.read_text(encoding="utf-8").splitlines()
            new_lines = []
            file_changed = False

            for line in lines:
                parts = line.strip().split()

                if len(parts) < 5:
                    new_lines.append(line)
                    continue

                if parts[0] != str(class_id):
                    file_changed = True
                    changed_objects += 1

                parts[0] = str(class_id)
                new_lines.append(" ".join(parts))

            if file_changed:
                label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                changed_files += 1

        messagebox.showinfo(
            "Batch Completed",
            f"Η μαζική αλλαγή ολοκληρώθηκε.\n\n"
            f"Εύρος εικόνων: {start} έως {end}\n"
            f"Νέα class: {class_id}: {class_name}\n"
            f"Labels που άλλαξαν: {changed_files}\n"
            f"Objects που άλλαξαν: {changed_objects}\n"
            f"Missing labels: {missing_labels}"
        )

        self.load_current_image()

    def choose_copy_target_folder(self):
        folder = filedialog.askdirectory(title="Επίλεξε φάκελο προορισμού για export")
        if folder:
            self.copy_target_var.set(folder)

    def copy_current_pair_to_folder(self):
        target_root_text = self.copy_target_var.get().strip()

        if not target_root_text:
            messagebox.showwarning("Warning", "Επίλεξε πρώτα φάκελο προορισμού.")
            return

        target_root = Path(target_root_text)

        if not target_root.exists():
            try:
                target_root.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Δεν μπόρεσε να δημιουργηθεί ο φάκελος προορισμού:\n{e}")
                return

        image_path = self.current_image_path
        label_path = self.current_label_path
        split = self.current_split

        if not image_path.exists():
            messagebox.showerror("Error", f"Δεν βρέθηκε η εικόνα:\n{image_path}")
            return

        self.save_annotations(show_popup=False)

        if not label_path.exists():
            messagebox.showwarning("Warning", f"Δεν βρέθηκε txt label για την εικόνα:\n{label_path}")
            return

        dest_images_dir = target_root / split / "images"
        dest_labels_dir = target_root / split / "labels"
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        dest_labels_dir.mkdir(parents=True, exist_ok=True)

        dest_image = dest_images_dir / image_path.name
        dest_label = dest_labels_dir / label_path.name

        try:
            shutil.copy2(image_path, dest_image)
            shutil.copy2(label_path, dest_label)
        except Exception as e:
            messagebox.showerror("Error", f"Δεν μπόρεσε να γίνει αντιγραφή:\n{e}")
            return

        messagebox.showinfo(
            "Copied",
            f"Το ζεύγος image+txt αντιγράφηκε επιτυχώς:\n\n"
            f"Image:\n{dest_image}\n\n"
            f"Label:\n{dest_label}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnnotationEditor(root)
    root.mainloop()
