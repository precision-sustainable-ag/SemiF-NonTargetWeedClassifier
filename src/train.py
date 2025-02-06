import logging
from ultralytics import YOLO

log = logging.getLogger(__name__)

def main(cfg):
    # Load a model
    task_config = cfg['train']
    model = YOLO(task_config['pretrained_model_path'])  # load a pretrained model (recommended for training)
    
    log.info(f"Started training script")
    
    
    yolo_logger = logging.getLogger("ultralytics")
    for handler in yolo_logger.handlers:
        log.addHandler(handler)
    
    # Train the model
    results = model.train(
        data=task_config['data'],
        epochs=int(task_config['epochs']),
        patience=int(task_config['patience']),
        batch=int(task_config['batch']),
        imgsz=int(task_config['imgsz']),
        workers=int(task_config['workers']),
        project=task_config['project'],
        name=task_config['name'],
        # hsv_h=0.000,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # degrees=0.0,
        # translate=0.0,
        # scale=0.0,
        # shear=0,
        # perspective=0.00,
        # flipud=0.5,
        # fliplr=0.5,
        # bgr=0.0,
        # mosaic=0.05,
        # mixup=0.05,
        # copy_paste=0.05,
        # copy_paste_mode="mix",
        # auto_augment=None,
        # erasing=0.0,
        # crop_fraction=0.10,
        )
    
    for handler in yolo_logger.handlers:
        log.removeHandler(handler)