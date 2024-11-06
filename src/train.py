from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="data",
    epochs=100,
    patience=20,
    batch=8,
    imgsz=128,
    workers=10,
    project="runs/classify/MD_covers",
    name="batch8_imgsz128_1392_n",
    # hsv_h=0.015,
    # hsv_s=0.7,
    # hsv_v=0.4,
    # degrees=0.90,
    # translate=0.51,
    # scale=0.0,
    # shear=10,
    # perspective=0.0001,
    # flipud=0.5,
    # fliplr=0.5,
    # bgr=0.15,
    # mosaic=0.15,
    # mixup=0.15,
    # copy_paste=0.15,
    # copy_paste_mode="mix",
    # auto_augment="randaugment",
    # erasing=0.0,
    # crop_fraction=0.10,
    )