from ultralytics import YOLO

# Load a model
name = "batch32_imgsz128_5468_n"
model = YOLO(f"runs/classify/NC_covers_grasses_binary/{name}/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(
    data="data",
    imgsz=128,
    batch=32,
    project="runs/classify/NC_covers_grasses_binary/val",
    name=f"{name}"
)  # no arguments needed, dataset and settings remembered