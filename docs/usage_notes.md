# Usage Notes

This document provides practical usage notes for the supporting GUI tools and Python scripts included in this repository. The tools were developed to support dataset curation, annotation review, synthetic dataset preparation, and qualitative evaluation for YOLO-format computer vision datasets.

## General Purpose

The tools in this repository are intended to support the following tasks:

* visual inspection of image-label pairs;
* review and correction of YOLO annotations;
* class distribution analysis;
* dataset reorganization;
* preparation of reduced, two-class, or one-class datasets;
* review and selection of synthetic images;
* support for real-plus-synthetic training set preparation;
* export of selected prediction examples for qualitative evaluation.

The tools are designed mainly for research and experimental workflows, not for fully automated operational deployment.

## Recommended Dataset Structure

Most tools assume that the dataset follows a YOLO-style folder structure:

```text
dataset/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

The `images/` folders should contain image files such as:

```text
.jpg
.jpeg
.png
.bmp
```

The `labels/` folders should contain YOLO annotation files with the same base filename as the corresponding image.

Example:

```text
train/images/example_001.jpg
train/labels/example_001.txt
```

## YOLO Label Format

Each YOLO label file should contain one row per annotated object:

```text
class_id x_center y_center width height
```

All coordinates must be normalized between 0 and 1.

Example:

```text
3 0.5124 0.4781 0.2310 0.1642
```

In this example:

* `3` is the class ID;
* `0.5124` is the normalized x-coordinate of the bounding box center;
* `0.4781` is the normalized y-coordinate of the bounding box center;
* `0.2310` is the normalized bounding box width;
* `0.1642` is the normalized bounding box height.

Empty `.txt` label files may be used for negative images, meaning images that do not contain any object of interest.

## Example `data.yaml`

A typical `data.yaml` file may follow this structure:

```yaml
path: ./examples/sample_dataset_structure
train: train/images
val: valid/images
test: test/images

names:
  0: CIV_VEHICLE
  1: MIL_ARM_VEHICLE
  2: MIL_ARTILLERY
  3: MIL_MBT
  4: MIL_MISS_LAUNCHER
  5: MIL_VEHICLE
```

The class IDs in the label files must match the class IDs defined in `data.yaml`.

## Recommended Workflow

A typical dataset preparation workflow may include the following steps:

1. Organize the dataset in YOLO format.
2. Check that every image has a corresponding label file.
3. Check that every label file has a corresponding image.
4. Review the dataset visually using the annotation editor.
5. Correct wrong class IDs or problematic bounding boxes.
6. Remove or reject images with very low quality or ambiguous content.
7. Use the dataset manager to inspect class distribution.
8. Create reduced, two-class, or one-class dataset versions if needed.
9. Review synthetic images before adding them to the training set.
10. Keep validation and test sets real whenever the goal is to evaluate real-world generalization.
11. Train the selected object detection model.
12. Export selected prediction examples for qualitative evaluation.

## Tool Usage

### Running a GUI Tool

From the root folder of the repository, run:

```bash
python tools/yolo_dataset_manager.py
```

or:

```bash
python tools/yolo_annotation_editor.py
```

The exact filename depends on the final names of the scripts included in the `tools/` folder.

Some tools may allow the user to select the dataset folder through a graphical interface. Other tools may assume that the dataset is located in the same folder as the script or in a predefined relative path.

## Path Handling

For portability, the tools should avoid hardcoded local paths such as:

```text
C:\Users\...
/content/drive/MyDrive/...
```

A recommended approach is to use relative paths based on the script location:

```python
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR
```

This allows the scripts to run more reliably when moved to another computer or folder.

## Synthetic Dataset Preparation

Synthetic images should not be added blindly to the training set. They should first be reviewed for:

* visual quality;
* object visibility;
* realism;
* annotation correctness;
* class consistency;
* relevance to the target detection problem;
* excessive similarity or duplication;
* mismatch with the real dataset domain.

Synthetic data should be treated as a complementary source of training data, not as a full replacement for curated real data.

## Real-Plus-Synthetic Training Sets

When creating real-plus-synthetic datasets, a recommended practice is:

* keep the validation set real;
* keep the test set real;
* add synthetic images only to the training set;
* document the number of real and synthetic images used;
* keep the class mapping consistent across real and synthetic data;
* avoid adding low-quality synthetic images that may increase domain gap.

This approach helps evaluate whether synthetic data improves generalization to real images.

## One-Class Dataset Preparation

For one-class experiments, such as a single-class `MIL_MBT` dataset, the target class should be remapped to class ID `0`.

Example:

```text
MIL_MBT → 0
```

All other class annotations should be removed.

Images that no longer contain the target class may be kept with empty label files. These images can function as negative examples or hard negatives, helping the model reduce false positives.

## Two-Class Dataset Preparation

For two-class experiments, only the selected classes should be retained.

Example:

```text
0: MIL_ARM_VEHICLE
1: MIL_MBT
```

All other class annotations should be removed. The `data.yaml` file must also be updated so that the class IDs match the new two-class structure.

## Quality Control Checklist

Before training, it is recommended to verify the following:

* All image files open correctly.
* Every image has a corresponding label file.
* Every label file has a corresponding image.
* Class IDs in labels match the `data.yaml` file.
* Bounding boxes are inside image boundaries.
* Empty labels are intentional.
* No unwanted duplicate images exist.
* No irrelevant files are inside the dataset folders.
* The dataset split is correct.
* The validation and test sets are not contaminated with synthetic images unless this is intentional.
* The dataset version is clearly named and documented.

## Prediction Export

Prediction export scripts may be used to generate qualitative examples for reports, appendices, or manual inspection.

A typical use case is:

```bash
python tools/make_individual_predictions_for_appendix.py
```

The script may require:

* a trained model file;
* a folder containing input images;
* an output folder for prediction images;
* optional confidence or IoU thresholds.

Trained model weights are not included in this repository.

## Data Availability

This repository does not include the real or synthetic datasets used in the thesis experiments.

The following materials are intentionally excluded:

* real military image datasets;
* synthetic image datasets used in experiments;
* trained model weights;
* Google Drive folders;
* training runs;
* large result folders;
* `.pt`, `.onnx`, or other model export files;
* large archives such as `.zip` or `.rar`.

Only code, documentation, screenshots, and minimal structural examples are provided.

## Security and Ethical Considerations

Users should ensure that any dataset processed with these tools complies with applicable legal, ethical, institutional, and security requirements.

The tools are intended for research, educational, and experimental dataset preparation workflows. They should not be used with restricted, classified, or sensitive data unless the user has the appropriate authorization and follows the relevant institutional procedures.

## Reproducibility Notes

For better reproducibility, it is recommended to document:

* dataset version;
* class mapping;
* number of images per split;
* number of objects per class;
* preprocessing steps;
* curation decisions;
* synthetic data source and selection criteria;
* training configuration;
* model version;
* evaluation metrics;
* hardware environment.

Keeping this information organized makes it easier to compare experiments and interpret model performance.

## Limitations

The tools are intended to support dataset preparation and inspection, but they do not replace expert judgment. Final decisions about image quality, annotation correctness, class definitions, and dataset suitability should be made by the researcher.

The tools may also require adaptation depending on the operating system, Python version, dataset size, folder structure, or specific experimental requirements.

## Suggested Use

The recommended use of this repository is as a supporting toolkit for research workflows involving YOLO-format datasets, especially when the goal is to evaluate how dataset curation and synthetic enrichment affect object detection performance.

