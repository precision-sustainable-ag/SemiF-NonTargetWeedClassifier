from ultralytics import YOLO
from pathlib import Path
import shutil
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime

class ImageBatchProcessor:
    def __init__(self, image_folder, batch_prefix, start_date, end_date):
        self.image_folder = Path(image_folder)
        self.batch_prefix = batch_prefix
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.df = None

    def load_batches(self, sample_size=10):
        # batches = [x for x in self.image_folder.glob("*") if self.batch_prefix in x.stem]
        batches = [x for x in self.image_folder.glob('*') if self.batch_prefix in x.stem and (self.start_date <= pd.to_datetime(x.stem.split('_')[1]) <= self.end_date)]
        if sample_size >= len(batches):
            sample_batches = batches
        else:
            sample_batches = random.sample(batches, sample_size)
        
        dfs = []
        
        for batch in tqdm(sample_batches):
            df = pd.read_csv(batch / f"{batch.stem}.csv")
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        self.filter_data()

    def filter_data(self):
        self.df["batch_date"] = pd.to_datetime(self.df["batch_id"].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
        self.df = self.df[(self.df["batch_date"] >= self.start_date) & (self.df["batch_date"] <= self.end_date)]
        self.df = self.df[self.df["common_name"] != "unknown"]

    def get_data(self):
        return self.df


class WeedPredictor:
    def __init__(self, model_path, label_map):
        self.model = YOLO(model_path)
        self.label_map = label_map

    def predict(self, img_path):
        results = self.model(img_path, imgsz=128)
        if len(results) > 1:
            print(f"Multiple results found for {img_path}")
            return None, None
        return results[0].probs.top1, results[0].probs.top1conf.cpu().item()

    def batch_predict(self,images, batch_size):
        def generate_batches(images):
            for i in range(0, len(images), batch_size):
                yield images[i:i+batch_size]

        def yield_predictions(batch, batch_predictions):
            for img_path, prediction in zip(batch, batch_predictions):
                yield prediction.probs.top1, prediction.probs.top1conf.cpu().item()
        
        results = []
        for batch in generate_batches(images):
            batch_predictions = self.model(source=batch, imgsz=128)
            
            results.extend(yield_predictions(batch, batch_predictions))
            # results.extend(self.model.predict(batch))
        return results

class PredictionSaver:
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)

    def save_prediction(self, img_path, confidence, target_class):
        if confidence > 0.95:
            return

        conf_ranges = [(0.85, 0.95), (0.65, 0.85), (0.5, 0.65), (0.35, 0.5), (0.15, 0.35), (0, 0.15)]
        for lower, upper in conf_ranges:
            if lower <= confidence < upper:
                target_type = "non_target" if target_class == "non_target" else "target"
                output_dest = self.output_folder / f"{target_type}_class_{int(lower*100)}_{int(upper*100)}"
                output_dest.mkdir(exist_ok=True, parents=True)
                shutil.copy2(img_path, output_dest)
                break


class BatchInferencePipeline:
    def __init__(self, image_folder, batch_prefix, model_path, label_map, output_folder, date_range, parallel_processing):
        self.processor = ImageBatchProcessor(image_folder, batch_prefix, *date_range)
        self.predictor = WeedPredictor(model_path, label_map)
        self.saver = PredictionSaver(output_folder)
        self.predicted_rows = []
        self.parallel_processing = parallel_processing

    def run(self):
        self.processor.load_batches()
        df = self.processor.get_data()
        

        if self.parallel_processing:
            images = []
            df['img_path'] = df.apply(lambda row: self.processor.image_folder / row['batch_id'] / f"{row['cutout_id']}.jpg", axis=1)
            images = df['img_path'].tolist()
            results = self.predictor.batch_predict(images, 512)
            for record, result in zip(df.iterrows(), results):
                row = record[1]
                target_class, confidence = result
                if target_class is None:
                    continue
                target_class_label = self.predictor.label_map[target_class]
                row["PredictedTargetWeed"] = True if target_class_label != "non_target" else False
                
                if confidence is not None:
                    self.saver.save_prediction(row['img_path'], confidence, target_class_label)
                    
                self.predicted_rows.append(row)
        else:
            for _, row in df.iterrows():
                img_name = f"{row['cutout_id']}.jpg"
                batch_id = row["batch_id"]
                img_path = self.processor.image_folder / batch_id / img_name
                
                target_class, confidence = self.predictor.predict(img_path)
                if target_class is None:
                    continue

                target_class_label = self.predictor.label_map[target_class]
                row["PredictedTargetWeed"] = True if target_class_label != "non_target" else False
                
                if confidence is not None:
                    self.saver.save_prediction(img_path, confidence, target_class_label)
                
                self.predicted_rows.append(row)
        
        self.save_results()

    def save_results(self):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_filename = self.saver.output_folder / f"prediction_batches_{timestamp}.csv"
        pd.DataFrame(self.predicted_rows).to_csv(output_filename, index=False)


# Usage example
pipeline = BatchInferencePipeline(
    image_folder="/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts",
    batch_prefix="MD",
    model_path="runs/classify/MD_covers/batch8_imgsz128_1030_n/weights/best.pt",
    label_map={
        0: "broadleaf",
        1: "grass",
        2: "hairy_vetch",
        3: "non_target"
    },
    output_folder="predictions_test",
    date_range=('2023-09-18', '2024-05-16'),
    parallel_processing=True
    # date_range=("2022-10-12", "2023-05-20")

)
start_time = datetime.now()
pipeline.run()
print(f"DONE. Time: {datetime.now() - start_time}")