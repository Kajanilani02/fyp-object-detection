from ultralytics import YOLO, RTDETR

def evaluate_model(model_weight_file, alias):
    if "rtdetr" in model_weight_file:
        model = RTDETR(model_weight_file)
    else:
        model = YOLO(model_weight_file)

    metrics = model.val(
        data="/kaggle/working/dataset/data.yaml",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.5,
        project="FYP_Object_Detection",
        name=f"Validation_{alias}"
    )

    print(f"Model: {alias}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

# Evaluate all three models
evaluate_model("yolo11n.pt", "YOLO11n")
evaluate_model("yolov8s.pt", "YOLOv8s")
evaluate_model("rtdetr-l.pt", "RT-DETR-l")
