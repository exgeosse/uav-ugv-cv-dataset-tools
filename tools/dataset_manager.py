from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import yaml
import shutil
from datetime import datetime
import csv
import time
import re

# =====================================================
# ΡΥΘΜΙΣΕΙΣ
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CREATE_BACKUP = True

# =====================================================


class YOLODatasetManager:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Dataset Manager - Analytics & Class Merge")

        self.dataset_dir = DATASET_DIR
        self.yaml_path = self.find_yaml()
        self.class_names = self.load_classes()

        self.log_file = self.dataset_dir / "class_merge_log.csv"

        self.setup_ui()
        self.refresh_analytics()

    def find_yaml(self):
        for name in ["data.yaml", "data.yml"]:
            path = self.dataset_dir / name
            if path.exists():
                return path

        raise FileNotFoundError("Δεν βρέθηκε data.yaml ή data.yml")

    def load_classes(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", {})

        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}

        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

        raise ValueError("Δεν βρέθηκαν σωστά classes στο YAML.")

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="YOLO Dataset Analytics & Class Manager",
            font=("Arial", 14, "bold")
        )
        title.pack(anchor="w", pady=5)

        self.dataset_label = ttk.Label(main, text=f"Dataset: {self.dataset_dir}")
        self.dataset_label.pack(anchor="w")

        self.summary_label = ttk.Label(main, text="", font=("Arial", 11, "bold"))
        self.summary_label.pack(anchor="w", pady=8)

        columns = ("class_id", "class_name", "objects", "images")
        self.tree = ttk.Treeview(main, columns=columns, show="headings", height=15)

        self.tree.heading("class_id", text="Class ID")
        self.tree.heading("class_name", text="Class Name")
        self.tree.heading("objects", text="Objects")
        self.tree.heading("images", text="Images containing class")

        self.tree.column("class_id", width=80)
        self.tree.column("class_name", width=260)
        self.tree.column("objects", width=120)
        self.tree.column("images", width=180)

        self.tree.pack(fill=tk.BOTH, expand=True, pady=8)

        split_frame = ttk.LabelFrame(main, text="Split Analytics")
        split_frame.pack(fill=tk.X, pady=8)

        self.split_label = ttk.Label(split_frame, text="")
        self.split_label.pack(anchor="w", padx=8, pady=6)

        merge_frame = ttk.LabelFrame(main, text="Change / Merge Entire Class")
        merge_frame.pack(fill=tk.X, pady=8)

        ttk.Label(merge_frame, text="Από class:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(merge_frame, text="Σε class:").grid(row=1, column=0, padx=5, pady=5, sticky="w")

        combo_values = [
            f"{cid}: {name}"
            for cid, name in sorted(self.class_names.items())
        ]

        self.from_class_var = tk.StringVar()
        self.to_class_var = tk.StringVar()

        self.from_combo = ttk.Combobox(
            merge_frame,
            textvariable=self.from_class_var,
            values=combo_values,
            state="readonly",
            width=40
        )
        self.from_combo.grid(row=0, column=1, padx=5, pady=5)

        self.to_combo = ttk.Combobox(
            merge_frame,
            textvariable=self.to_class_var,
            values=combo_values,
            state="readonly",
            width=40
        )
        self.to_combo.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(
            merge_frame,
            text="Preview Change",
            command=self.preview_change
        ).grid(row=0, column=2, padx=8)

        ttk.Button(
            merge_frame,
            text="Apply Change",
            command=self.apply_class_change
        ).grid(row=1, column=2, padx=8)

        yaml_frame = ttk.LabelFrame(main, text="Class Names / YAML Editor")
        yaml_frame.pack(fill=tk.X, pady=8)

        ttk.Button(
            yaml_frame,
            text="Rename Selected Class",
            command=self.rename_selected_class
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            yaml_frame,
            text="Add New Class",
            command=self.add_new_class
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            yaml_frame,
            text="Delete Selected Class",
            command=self.delete_selected_class
        ).pack(side=tk.LEFT, padx=5, pady=5)

        rename_frame = ttk.LabelFrame(main, text="Rename ALL jpg+txt pairs")
        rename_frame.pack(fill=tk.X, pady=8)

        ttk.Label(rename_frame, text="Base Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.rename_base_var = tk.StringVar(value="MILDATA")
        self.rename_base_entry = ttk.Entry(rename_frame, textvariable=self.rename_base_var, width=30)
        self.rename_base_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.rename_add_split_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rename_frame,
            text="Add Split Name",
            variable=self.rename_add_split_var
        ).grid(row=0, column=2, padx=8, pady=5, sticky="w")

        self.rename_backup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rename_frame,
            text="Backup Before Rename",
            variable=self.rename_backup_var
        ).grid(row=0, column=3, padx=8, pady=5, sticky="w")

        ttk.Button(
            rename_frame,
            text="Preview Rename",
            command=self.preview_rename_pairs
        ).grid(row=0, column=4, padx=8, pady=5)

        ttk.Button(
            rename_frame,
            text="Apply Rename",
            command=self.apply_rename_pairs
        ).grid(row=0, column=5, padx=8, pady=5)

        ttk.Label(
            rename_frame,
            text="Format: <BASE>_<SPLIT>_<TIMESTAMP_NS>.jpg + .txt"
        ).grid(row=1, column=0, columnspan=6, padx=5, pady=(0, 5), sticky="w")

        ttk.Button(
            main,
            text="Refresh Analytics",
            command=self.refresh_analytics
        ).pack(anchor="e", pady=6)

    def get_split_dirs(self):
        result = []

        for split in SPLITS:
            images_dir = self.dataset_dir / split / "images"
            labels_dir = self.dataset_dir / split / "labels"

            if images_dir.exists() and labels_dir.exists():
                result.append((split, images_dir, labels_dir))

        for split in SPLITS:
            images_dir = self.dataset_dir / "images" / split
            labels_dir = self.dataset_dir / "labels" / split

            if images_dir.exists() and labels_dir.exists():
                result.append((split, images_dir, labels_dir))

        return result

    def find_image_for_label(self, images_dir, stem):
        for ext in IMAGE_EXTENSIONS:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def read_label_classes(self, label_path):
        classes = []

        if not label_path.exists():
            return classes

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                classes.append(int(float(parts[0])))
            except ValueError:
                continue

        return classes

    def refresh_analytics(self):
        self.class_names = self.load_classes()

        object_counter = Counter()
        image_counter = Counter()
        split_image_counter = Counter()
        split_label_counter = Counter()
        split_object_counter = Counter()

        total_images = 0
        total_labels = 0
        total_objects = 0
        empty_labels = 0

        for split, images_dir, labels_dir in self.get_split_dirs():
            images = [
                p for p in images_dir.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            labels = list(labels_dir.rglob("*.txt"))

            split_image_counter[split] += len(images)
            split_label_counter[split] += len(labels)

            total_images += len(images)
            total_labels += len(labels)

            for label_path in labels:
                classes = self.read_label_classes(label_path)

                if not classes:
                    empty_labels += 1
                    continue

                total_objects += len(classes)
                split_object_counter[split] += len(classes)

                unique_classes = set(classes)

                for class_id in classes:
                    object_counter[class_id] += 1

                for class_id in unique_classes:
                    image_counter[class_id] += 1

        self.summary_label.config(
            text=(
                f"Images: {total_images} | "
                f"Labels: {total_labels} | "
                f"Objects: {total_objects} | "
                f"Classes: {len(self.class_names)} | "
                f"Empty labels: {empty_labels}"
            )
        )

        split_lines = []
        for split in SPLITS:
            if split in split_image_counter or split in split_label_counter:
                split_lines.append(
                    f"{split}: "
                    f"images={split_image_counter[split]}, "
                    f"labels={split_label_counter[split]}, "
                    f"objects={split_object_counter[split]}"
                )

        self.split_label.config(text="\n".join(split_lines))

        for row in self.tree.get_children():
            self.tree.delete(row)

        for class_id, class_name in sorted(self.class_names.items()):
            self.tree.insert(
                "",
                tk.END,
                values=(
                    class_id,
                    class_name,
                    object_counter[class_id],
                    image_counter[class_id]
                )
            )

        self.object_counter = object_counter
        self.image_counter = image_counter

    def get_selected_ids(self):
        if not self.from_class_var.get() or not self.to_class_var.get():
            messagebox.showwarning("Warning", "Επίλεξε και τις δύο classes.")
            return None, None

        from_id = int(self.from_class_var.get().split(":")[0])
        to_id = int(self.to_class_var.get().split(":")[0])

        if from_id == to_id:
            messagebox.showwarning("Warning", "Η αρχική και η τελική class είναι ίδιες.")
            return None, None

        return from_id, to_id

    def preview_change(self):
        from_id, to_id = self.get_selected_ids()

        if from_id is None:
            return

        count = self.object_counter.get(from_id, 0)
        images = self.image_counter.get(from_id, 0)

        messagebox.showinfo(
            "Preview",
            f"Θα αλλάξουν:\n\n"
            f"Class {from_id}: {self.class_names[from_id]}\n"
            f"σε\n"
            f"Class {to_id}: {self.class_names[to_id]}\n\n"
            f"Objects που θα επηρεαστούν: {count}\n"
            f"Εικόνες που περιέχουν την class: {images}"
        )

    def backup_file(self, path):
        if not CREATE_BACKUP or not path.exists():
            return

        backup_dir = self.dataset_dir / "_class_merge_backup"
        rel = path.relative_to(self.dataset_dir)
        dest = backup_dir / rel

        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            shutil.copy2(path, dest)

    def apply_class_change(self):
        from_id, to_id = self.get_selected_ids()

        if from_id is None:
            return

        confirm = messagebox.askyesno(
            "Confirm",
            f"Είσαι σίγουρος ότι θέλεις να αλλάξεις ΟΛΗ την class:\n\n"
            f"{from_id}: {self.class_names[from_id]}\n\n"
            f"σε:\n\n"
            f"{to_id}: {self.class_names[to_id]}\n\n"
            f"Αυτό θα τροποποιήσει όλα τα σχετικά .txt labels."
        )

        if not confirm:
            return

        changed_files = 0
        changed_objects = 0

        for split, images_dir, labels_dir in self.get_split_dirs():
            for label_path in labels_dir.rglob("*.txt"):
                lines = label_path.read_text(encoding="utf-8").splitlines()
                new_lines = []
                file_changed = False

                for line in lines:
                    parts = line.strip().split()

                    if len(parts) < 5:
                        new_lines.append(line)
                        continue

                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        new_lines.append(line)
                        continue

                    if class_id == from_id:
                        parts[0] = str(to_id)
                        file_changed = True
                        changed_objects += 1

                    new_lines.append(" ".join(parts))

                if file_changed:
                    self.backup_file(label_path)
                    label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    changed_files += 1

        self.write_log(from_id, to_id, changed_files, changed_objects)

        messagebox.showinfo(
            "Completed",
            f"Η αλλαγή ολοκληρώθηκε.\n\n"
            f"Labels που τροποποιήθηκαν: {changed_files}\n"
            f"Objects που άλλαξαν class: {changed_objects}"
        )

        self.refresh_analytics()
        self.refresh_combo_values()

    def save_yaml_classes(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
           data = yaml.safe_load(f)

        data["names"] = {
            int(cid): name
            for cid, name in sorted(self.class_names.items())
            }

        data["nc"] = len(self.class_names)

        self.backup_file(self.yaml_path)

        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


    def refresh_combo_values(self):
        combo_values = [
            f"{cid}: {name}"
            for cid, name in sorted(self.class_names.items())
        ]

        self.from_combo["values"] = combo_values
        self.to_combo["values"] = combo_values


    def get_selected_tree_class_id(self):
        selected = self.tree.selection()

        if not selected:
            messagebox.showwarning("Warning", "Επίλεξε πρώτα μία class από τον πίνακα.")
            return None

        values = self.tree.item(selected[0], "values")
        return int(values[0])


    def rename_selected_class(self):
        class_id = self.get_selected_tree_class_id()

        if class_id is None:
            return

        old_name = self.class_names[class_id]

        new_name = simpledialog.askstring(
            "Rename Class",
            f"Νέο όνομα για την class {class_id}: {old_name}",
            initialvalue=old_name
        )

        if not new_name:
            return

        new_name = new_name.strip()

        if not new_name:
            return

        self.class_names[class_id] = new_name
        self.save_yaml_classes()
        self.refresh_analytics()
        self.refresh_combo_values()

        messagebox.showinfo(
            "Completed",
            f"Η class {class_id} μετονομάστηκε σε:\n{new_name}"
        )


    def add_new_class(self):
        new_name = simpledialog.askstring(
            "Add New Class",
            "Όνομα νέας class:"
        )

        if not new_name:
            return

        new_name = new_name.strip()

        if not new_name:
            return

        existing_names = set(self.class_names.values())

        if new_name in existing_names:
            messagebox.showwarning("Warning", "Υπάρχει ήδη class με αυτό το όνομα.")
            return

        new_id = max(self.class_names.keys()) + 1 if self.class_names else 0

        self.class_names[new_id] = new_name
        self.save_yaml_classes()
        self.refresh_analytics()
        self.refresh_combo_values()

        messagebox.showinfo(
            "Completed",
            f"Προστέθηκε νέα class:\n\n{new_id}: {new_name}"
        )

    def find_image_for_label_any_structure(self, label_path):
        stem = label_path.stem

        # dataset/train/labels/img.txt -> dataset/train/images/img.jpg
        if label_path.parent.name == "labels":
            images_dir = label_path.parent.parent / "images"

            for ext in IMAGE_EXTENSIONS:
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    return p

        # dataset/labels/train/img.txt -> dataset/images/train/img.jpg
        if label_path.parent.parent.name == "labels":
            split = label_path.parent.name
            images_dir = label_path.parent.parent.parent / "images" / split

            for ext in IMAGE_EXTENSIONS:
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    return p

        return None

    def delete_selected_class(self):
        class_id_to_delete = self.get_selected_tree_class_id()

        if class_id_to_delete is None:
            return

        class_name_to_delete = self.class_names[class_id_to_delete]

        confirm = messagebox.askyesno(
            "Confirm Delete Class",
            f"Θέλεις σίγουρα να διαγράψεις την class:\n\n"
            f"{class_id_to_delete}: {class_name_to_delete}\n\n"
            f"Θα γίνουν τα εξής:\n"
            f"- Θα διαγραφούν jpg+txt που έχουν ΜΟΝΟ αυτή την class\n"
            f"- Θα αφαιρεθεί αυτή η class από mixed labels\n"
            f"- Θα γίνει re-index των υπόλοιπων classes\n"
            f"- Θα ενημερωθεί το data.yaml\n\n"
            f"Συνέχεια;"
        )

        if not confirm:
            return

        old_class_names = dict(self.class_names)

        keep_old_ids = [
            cid for cid in sorted(old_class_names.keys())
            if cid != class_id_to_delete
        ]

        old_to_new = {
            old_id: new_id
            for new_id, old_id in enumerate(keep_old_ids)
        }

        new_class_names = {
            old_to_new[old_id]: old_class_names[old_id]
            for old_id in keep_old_ids
        }

        changed_label_files = 0
        removed_objects = 0
        deleted_pairs = 0
        checked_labels = 0

        for split, images_dir, labels_dir in self.get_split_dirs():
            for label_path in labels_dir.rglob("*.txt"):
                checked_labels += 1

                lines = label_path.read_text(encoding="utf-8").splitlines()
                new_lines = []

                for line in lines:
                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    try:
                        old_class_id = int(float(parts[0]))
                    except ValueError:
                        continue

                    if old_class_id == class_id_to_delete:
                        removed_objects += 1
                        continue

                    if old_class_id not in old_to_new:
                        continue

                    parts[0] = str(old_to_new[old_class_id])
                    new_lines.append(" ".join(parts))

                image_path = self.find_image_for_label_any_structure(label_path)

                # Αν μετά την αφαίρεση δεν έμεινε κανένα annotation,
                # τότε διαγράφεται το ζευγάρι jpg+txt.
                if len(new_lines) == 0:
                    self.backup_file(label_path)

                    if image_path and image_path.exists():
                        self.backup_file(image_path)
                        image_path.unlink()

                    label_path.unlink(missing_ok=True)

                    deleted_pairs += 1
                    continue

                if new_lines != lines:
                    self.backup_file(label_path)
                    label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    changed_label_files += 1

        # Update YAML
        self.class_names = new_class_names
        self.save_yaml_classes()

        self.write_delete_class_log(
            class_id_to_delete,
            class_name_to_delete,
            checked_labels,
            changed_label_files,
            removed_objects,
            deleted_pairs
        )

        self.refresh_analytics()
        self.refresh_combo_values()

        messagebox.showinfo(
            "Completed",
            f"Η class διαγράφηκε επιτυχώς.\n\n"
            f"Class: {class_id_to_delete}: {class_name_to_delete}\n"
            f"Labels που ελέγχθηκαν: {checked_labels}\n"
            f"Labels που τροποποιήθηκαν: {changed_label_files}\n"
            f"Objects που αφαιρέθηκαν: {removed_objects}\n"
            f"Ζεύγη jpg+txt που διαγράφηκαν: {deleted_pairs}"
        )



    def sanitize_base_name(self, value):
        value = (value or "").strip()
        if not value:
            value = "DATASET"

        # Κρατάμε μόνο ασφαλείς χαρακτήρες για filenames.
        value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
        value = value.strip("_")

        return value or "DATASET"

    def find_image_for_label_in_dir(self, images_dir, stem):
        for ext in IMAGE_EXTENSIONS:
            image_path = images_dir / f"{stem}{ext}"
            if image_path.exists():
                return image_path
        return None

    def collect_image_label_pairs(self):
        pairs = []
        missing_labels = 0
        orphan_labels = 0

        for split, images_dir, labels_dir in self.get_split_dirs():
            image_paths = sorted(
                p for p in images_dir.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )

            image_stems = set()

            for image_path in image_paths:
                label_path = labels_dir / f"{image_path.stem}.txt"
                image_stems.add(image_path.stem)

                if label_path.exists():
                    pairs.append((split, image_path, label_path))
                else:
                    missing_labels += 1

            for label_path in labels_dir.rglob("*.txt"):
                if label_path.stem not in image_stems:
                    orphan_labels += 1

        return pairs, missing_labels, orphan_labels

    def build_rename_plan(self):
        base_name = self.sanitize_base_name(self.rename_base_var.get())
        add_split = self.rename_add_split_var.get()

        pairs, missing_labels, orphan_labels = self.collect_image_label_pairs()

        plan = []
        used_names = set()
        base_timestamp = time.time_ns()

        for counter, (split, image_path, label_path) in enumerate(pairs, start=1):
            timestamp_ns = base_timestamp + counter

            if add_split:
                new_stem = f"{base_name}_{split}_{timestamp_ns}"
            else:
                new_stem = f"{base_name}_{timestamp_ns}"

            # Πολύ σπάνιο, αλλά εξασφαλίζει μηδενικές συγκρούσεις μέσα στο plan.
            while new_stem in used_names:
                timestamp_ns += 1
                if add_split:
                    new_stem = f"{base_name}_{split}_{timestamp_ns}"
                else:
                    new_stem = f"{base_name}_{timestamp_ns}"

            used_names.add(new_stem)

            new_image_path = image_path.with_name(f"{new_stem}{image_path.suffix.lower()}")
            new_label_path = label_path.with_name(f"{new_stem}.txt")

            plan.append({
                "split": split,
                "old_image": image_path,
                "old_label": label_path,
                "new_image": new_image_path,
                "new_label": new_label_path,
            })

        return plan, missing_labels, orphan_labels

    def preview_rename_pairs(self):
        plan, missing_labels, orphan_labels = self.build_rename_plan()

        if not plan:
            messagebox.showinfo(
                "Preview Rename",
                "Δεν βρέθηκαν ζεύγη image+txt για μετονομασία."
            )
            return

        preview_lines = []
        for item in plan[:12]:
            preview_lines.append(
                f"{item['old_image'].name}  ->  {item['new_image'].name}"
            )

        if len(plan) > 12:
            preview_lines.append(f"... και ακόμη {len(plan) - 12} ζεύγη")

        messagebox.showinfo(
            "Preview Rename",
            f"Θα μετονομαστούν {len(plan)} ζεύγη image+txt.\n\n"
            f"Missing labels που δεν θα πειραχτούν: {missing_labels}\n"
            f"Orphan labels που δεν θα πειραχτούν: {orphan_labels}\n\n"
            f"Παράδειγμα:\n" + "\n".join(preview_lines)
        )

    def backup_for_rename(self, path):
        if not self.rename_backup_var.get() or not path.exists():
            return

        backup_dir = self.dataset_dir / "_rename_backup"
        rel = path.relative_to(self.dataset_dir)
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            shutil.copy2(path, dest)

    def write_rename_log(self, rows):
        log_file = self.dataset_dir / "rename_pairs_log.csv"
        file_exists = log_file.exists()

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "split",
                    "old_image",
                    "old_label",
                    "new_image",
                    "new_label"
                ])

            now = datetime.now().isoformat(timespec="seconds")
            for row in rows:
                writer.writerow([
                    now,
                    row["split"],
                    str(row["old_image"]),
                    str(row["old_label"]),
                    str(row["new_image"]),
                    str(row["new_label"]),
                ])

    def apply_rename_pairs(self):
        plan, missing_labels, orphan_labels = self.build_rename_plan()

        if not plan:
            messagebox.showinfo(
                "Apply Rename",
                "Δεν βρέθηκαν ζεύγη image+txt για μετονομασία."
            )
            return

        confirm = messagebox.askyesno(
            "Confirm Rename",
            f"Θέλεις σίγουρα να μετονομαστούν ΟΛΑ τα ζεύγη image+txt;\n\n"
            f"Ζεύγη που θα μετονομαστούν: {len(plan)}\n"
            f"Missing labels που δεν θα πειραχτούν: {missing_labels}\n"
            f"Orphan labels που δεν θα πειραχτούν: {orphan_labels}\n\n"
            f"Θα δημιουργηθεί log αρχείο: rename_pairs_log.csv"
        )

        if not confirm:
            return

        renamed_rows = []
        temp_moves = []

        try:
            # Backup πριν γίνει οποιαδήποτε μετονομασία.
            for item in plan:
                self.backup_for_rename(item["old_image"])
                self.backup_for_rename(item["old_label"])

            # Πρώτα μετονομάζουμε σε προσωρινά ονόματα ώστε να αποφύγουμε συγκρούσεις.
            for idx, item in enumerate(plan, start=1):
                old_image = item["old_image"]
                old_label = item["old_label"]

                tmp_image = old_image.with_name(f".__tmp_rename_{time.time_ns()}_{idx}{old_image.suffix.lower()}")
                tmp_label = old_label.with_name(f".__tmp_rename_{time.time_ns()}_{idx}.txt")

                old_image.rename(tmp_image)
                old_label.rename(tmp_label)

                temp_moves.append((item, tmp_image, tmp_label))

            # Μετά από τα προσωρινά στα τελικά ονόματα.
            for item, tmp_image, tmp_label in temp_moves:
                new_image = item["new_image"]
                new_label = item["new_label"]

                if new_image.exists() or new_label.exists():
                    raise FileExistsError(
                        f"Υπάρχει ήδη αρχείο προορισμού:\n{new_image}\nή\n{new_label}"
                    )

                tmp_image.rename(new_image)
                tmp_label.rename(new_label)
                renamed_rows.append(item)

        except Exception as e:
            messagebox.showerror(
                "Rename Error",
                f"Η μετονομασία σταμάτησε λόγω σφάλματος:\n\n{e}\n\n"
                f"Έλεγξε τον φάκελο dataset και το backup."
            )
            self.refresh_analytics()
            return

        self.write_rename_log(renamed_rows)
        self.refresh_analytics()

        messagebox.showinfo(
            "Rename Completed",
            f"Η μετονομασία ολοκληρώθηκε.\n\n"
            f"Ζεύγη που μετονομάστηκαν: {len(renamed_rows)}\n"
            f"Missing labels που δεν πειράχτηκαν: {missing_labels}\n"
            f"Orphan labels που δεν πειράχτηκαν: {orphan_labels}\n\n"
            f"Backup folder: _rename_backup\n"
            f"Log file: rename_pairs_log.csv"
        )

    def write_delete_class_log(
        self,
        deleted_id,
        deleted_name,
        checked_labels,
        changed_label_files,
        removed_objects,
        deleted_pairs
    ):
        log_file = self.dataset_dir / "class_delete_log.csv"

        file_exists = log_file.exists()

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "deleted_id",
                    "deleted_name",
                    "checked_labels",
                    "changed_label_files",
                    "removed_objects",
                    "deleted_pairs"
                ])

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                deleted_id,
                deleted_name,
                checked_labels,
                changed_label_files,
                removed_objects,
                deleted_pairs
            ])

    def write_log(self, from_id, to_id, changed_files, changed_objects):
        file_exists = self.log_file.exists()

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "from_id",
                    "from_class",
                    "to_id",
                    "to_class",
                    "changed_files",
                    "changed_objects"
                ])

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                from_id,
                self.class_names.get(from_id, ""),
                to_id,
                self.class_names.get(to_id, ""),
                changed_files,
                changed_objects
            ])


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODatasetManager(root)
    root.mainloop()