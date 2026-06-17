# Sample YOLO Dataset Structure

This folder provides a minimal example of the dataset structure expected by the supporting tools in this repository.

It does not contain real images, real annotations, military datasets, synthetic datasets, trained model weights, or experimental results. The purpose of this folder is only to demonstrate the expected organization of a YOLO-format dataset.

## Folder Structure

```text
sample_dataset_structure/
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

## YOLO Dataset Logic

In a YOLO-format object detection dataset, each image file should have a corresponding label file with the same base filename.

Example:

```text
train/images/example_001.jpg
train/labels/example_001.txt
```

The label file should contain one row per object:

```text
class_id x_center y_center width height
```

All bounding box coordinates must be normalized between 0 and 1.

## Empty Labels

An empty `.txt` label file may be used when an image does not contain any object of interest. Such images can function as negative examples or hard negatives during training.

## Class Mapping

The example `data.yaml` file uses the following six-class structure:

```text
0: CIV_VEHICLE
1: MIL_ARM_VEHICLE
2: MIL_ARTILLERY
3: MIL_MBT
4: MIL_MISS_LAUNCHER
5: MIL_VEHICLE
```

This class structure is provided only as an example and can be adapted depending on the target dataset and experiment.

## Important Note

No actual dataset is included in this folder. Users should place their own images and label files in the corresponding folders while ensuring that all data comply with applicable legal, ethical, institutional, and security requirements.
