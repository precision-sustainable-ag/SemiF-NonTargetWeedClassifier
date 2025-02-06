from ultralytics import YOLO
from pathlib import Path
# Load a model
name = "16_imgsz128_3500_n"
model_path = Path(f"runs/classify/TX_covers_multiclass/{name}/weights/best.pt")
data_dir = Path("data")

model = YOLO(model_path)  # load a custom model

# Validate the model
metrics = model.val(
    data=data_dir,
    imgsz=128,
    batch=16,
    project=model_path.parent.parent,
    name=f"val"
)  # no arguments needed, dataset and settings remembered