# UAV/UGV Computer Vision Dataset Curation Tools

This repository contains supporting GUI tools and Python scripts developed in the context of a diploma thesis on dataset curation and synthetic enrichment for computer vision applications in autonomous UAV and UGV systems.

The tools support YOLO-format dataset inspection, annotation review, class management, dataset reorganization, synthetic image review, augmentation workflow support, and prediction export for qualitative evaluation.

## Thesis Context

The tools were developed and used as part of the diploma thesis:

**Dataset Curation and Augmentation for Computer Vision in Autonomous UAV and UGV Systems**

The experimental workflow focused on object detection for military and civilian vehicle classes using YOLO11n and RT-DETR-L models. The main methodological focus was the effect of real dataset curation and synthetic dataset enrichment on object detection performance.

## Repository Structure

```text
tools/
  GUI tools and supporting Python scripts

screenshots/
  Example screenshots of the graphical tools

examples/
  Minimal example dataset structure

docs/
  Additional usage notes
```

## Tools

### 1. Annotation Editor

File:

```text
tools/annotation_editor.py
```

Purpose:

* Visual inspection of YOLO annotations
* Bounding box review
* Class label correction
* Image-by-image dataset curation
* Detection of problematic image-label pairs

### 2. Dataset Manager

File:

```text
tools/dataset_manager.py
```

Purpose:

* Class distribution analysis
* Dataset reorganization
* Class ID and class name checking
* Support for reduced, two-class, and one-class dataset preparation
* Dataset quality control before model training

### 3. Synthetic Capture / Review Tool

File:

```text
tools/synthetic_capture_review.py
```

Purpose:

* Review of synthetic images generated from simulated visual environments
* Selection and rejection of synthetic samples
* Support for controlled synthetic dataset construction
* Assistance in building real-plus-synthetic training sets

### 4. Synthetic Augmentation Studio

File:

```text
tools/synthetic_augmentation_studio.py
```

Purpose:

* Synthetic dataset enrichment workflow
* Preview and quality control of augmented samples
* Support for YOLO-format datasets
* Assistance in dataset preparation before training

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

Run a GUI tool from the repository root:

```bash
python tools/dataset_manager.py
```

Some tools may also allow the user to select a dataset folder through the graphical interface.

## Dataset Format

The tools assume a YOLO-style dataset structure:

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

Each YOLO label file follows the format:

```text
class_id x_center y_center width height
```

All bounding box coordinates are normalized between 0 and 1.

## Example `data.yaml`

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

## Important Notes

This repository does **not** include:

* Real military datasets
* Synthetic datasets used in the experiments
* Trained model weights
* Google Drive folders
* Experiment result folders
* Large archives or image collections

Only supporting code, documentation, screenshots, and minimal structural examples are provided.

## Data and Security Considerations

The tools were designed to support dataset preparation and experimental workflows. Any real or sensitive datasets used during the diploma thesis are not included in this repository. Users should ensure that any dataset processed with these tools complies with applicable legal, ethical, institutional, and security requirements.

## License

This repository is released under the MIT License unless otherwise stated.

## Citation

If you use or refer to this repository, you may cite it as:

```text
Exarchos, G. (2026). Supporting GUI tools for UAV/UGV computer vision dataset curation and synthetic enrichment [Source code]. GitHub.
```
