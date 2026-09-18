# fyp-object-detection
Object detection module for ground-level post-disaster damage assessment using YOLO11n, YOLOv8s, and RT-DETR-l
# Object Detection for Post-Disaster Damage Assessment

This repository contains the training and evaluation scripts for the object detection module of a ground-level post-disaster damage assessment system. Three architectures are benchmarked: YOLO11n, YOLOv8s, and RT-DETR-l.

## Dataset

- 1,349 ground-level disaster images
- Four classes: debris, landslide, structures, uprooted_tree
- 80/20 train-validation split

## Models

- YOLO11n (2.58M parameters)
- YOLOv8s (11.13M parameters)
- RT-DETR-l (31.99M parameters)

## Files

- `dataset_prepare.py` – Extracts dataset, performs 80/20 split, creates data.yaml
- `yolo11n_train.py` – Trains YOLO11n for 30 epochs at 640x640
- `yolov8s_train.py` – Trains YOLOv8s for 30 epochs at 640x640
- `rtdetr_train.py` – Trains RT-DETR-l for 30 epochs at 640x640
- `validate.py` – Evaluates trained models on validation set

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare dataset: `python dataset_prepare.py`
3. Train model: `python yolo11n_train.py` (or yolov8s/rtdetr)
4. Validate: `python validate.py`

## Results

- YOLO11n: Precision 0.913, mAP@50 0.884, 2.2 ms inference
- YOLOv8s: Precision 0.888, mAP@50 0.868, 5.5 ms inference
- RT-DETR-l: Precision 0.880, mAP@50 0.901, 17.9 ms inference

## Reference

This work is part of a final year project on AI-powered disaster damage assessment.
