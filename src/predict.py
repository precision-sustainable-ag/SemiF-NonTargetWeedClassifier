"""
This script performs batch predictions on image data leveraging a "teacher" classification model
to classify images into target and non-target categories. Predicted images are organized 
into specific confidence intervals, enabling the identification of low-confidence or challenging 
cases for distinguishing between target and non-target weeds.

The categorized images can later be validated and labeled using the image_labeler.py script. 
This approach is useful for dataset curation, as it allows a focused review 
of ambiguous or hard-to-classify cases, enhancing the dataset by identifying examples that 
are harder to predict and might need further attention for labeling, particularly for non-target weeds.
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import logging
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)


class ImageBatchProcessor:
    """
    A class to handle loading and filtering image batches based on specified date range
    and batch prefix.

    Attributes:
        image_folder (Path): Path to the folder containing image batches.
        batch_prefix (str): Prefix used to identify relevant batches.
        start_date (datetime): Start date for filtering batches.
        end_date (datetime): End date for filtering batches.
        sample_size (int): Number of batches to sample.
        df (DataFrame): DataFrame storing the concatenated image data from sampled batches.

    Methods:
        load_batches(): Loads and samples image batches based on the specified date range
            and batch prefix, concatenating the results into a DataFrame.
        get_data(): Returns the loaded and filtered DataFrame containing image data.
    """

    def __init__(self, image_folder, batch_prefix, start_date, end_date, sample_size):
        self.image_folder = Path(image_folder)
        self.batch_prefix = batch_prefix
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.df = None
        self.sample_size = sample_size

    def load_batches(self):
        log.info("Loading batches")
        batches = [x for x in self.image_folder.glob('*') 
                   if self.batch_prefix in x.stem and 
                   (self.start_date <= pd.to_datetime(x.stem.split('_')[1]) <= self.end_date)]
        if self.sample_size >= len(batches):
            sample_batches = batches
        else:
            sample_batches = random.sample(batches, self.sample_size)
        
        dfs = []
        
        for batch in tqdm(sample_batches):
            df = pd.read_csv(batch / f"{batch.stem}.csv")
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df[self.df["common_name"] != "unknown"]
        log.info(f"Working with {len(self.df)} images")
        
    def get_data(self):
        return self.df


class WeedPredictor:
    """
    A class to perform weed prediction using a "teacher" YOLO model.

    Attributes:
        model (YOLO): YOLO model instance for image prediction.
        label_map (dict): Dictionary mapping class indices to class names.

    Methods:
        predict(img_path): Predicts the class and confidence for a single image.
        batch_predict(images, batch_size): Predicts classes and confidences for a batch of images.
    """
    
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
            for _, prediction in zip(batch, batch_predictions):
                yield prediction.probs.top1, prediction.probs.top1conf.cpu().item()
        
        results = []
        for batch in generate_batches(images):
            batch_predictions = self.model(source=batch, imgsz=128)
            
            results.extend(yield_predictions(batch, batch_predictions))
            # results.extend(self.model.predict(batch))
        log.info("Predictions generated (batched mode)")
        return results

class PredictionSaver:
    """
    A class to save image predictions by organizing them based on confidence score ranges.

    Attributes:
        output_folder (Path): Path to the folder where predictions are saved.

    Methods:
        save_prediction(img_path, confidence, target_class): Saves the image prediction
            to a specific subfolder based on the confidence score and target class.
    """
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
    """
    Integrates image loading, prediction, and saving results.

    Attributes:
        processor (ImageBatchProcessor): Instance for loading and filtering image batches.
        predictor (WeedPredictor): Instance for predicting weed presence in images.
        saver (PredictionSaver): Instance for saving predictions based on confidence scores.
        predicted_rows (list): List to store prediction results for each image.
        parallel_processing (bool): Flag to enable parallel processing of predictions.
        batch_size (int): Batch size for predictions.
        sample_size (int): Number of batches to sample.

    Methods:
        run(): Executes the entire pipeline, loading images, predicting, and saving results.
        save_results(): Saves prediction results to a CSV file and returns the file path.
    """

    def __init__(self, image_folder, batch_prefix, model_path, label_map, output_folder, date_range, parallel_processing, batch_size, sample_size):
        self.processor = ImageBatchProcessor(image_folder, batch_prefix, *date_range, sample_size)
        self.predictor = WeedPredictor(model_path, label_map)
        self.saver = PredictionSaver(output_folder)
        self.predicted_rows = []
        self.parallel_processing = parallel_processing
        self.batch_size = batch_size

    def run(self):
        self.processor.load_batches()
        df = self.processor.get_data()

        if self.parallel_processing:
            log.info(f"Generating labels in parallel with batch size of {self.batch_size}, resulting in approximately { int(len(df)/self.batch_size) } batches")
            images = []
            df['img_path'] = df.apply(lambda row: self.processor.image_folder / row['batch_id'] / f"{row['cutout_id']}.jpg", axis=1)
            images = df['img_path'].tolist()
            results = self.predictor.batch_predict(images, self.batch_size)

            log.info("Saving predictions and diving images based on confidence scores (batched mode)")
            for record, result in zip(df.iterrows(), results):
                row = record[1]
                target_class, confidence = result
                if target_class is None:
                    continue
                target_class_label = self.predictor.label_map[target_class]
                row["PredictedTargetWeed"] = True if target_class_label != "non_target" else False
                row["PredictedTargetWeed_Confidence"] = confidence
                
                if confidence is not None:
                    self.saver.save_prediction(row['img_path'], confidence, target_class_label)
                    
                self.predicted_rows.append(row)
        else:
            log.info("Generating predictions (serial mode)")
            for _, row in df.iterrows():
                img_name = f"{row['cutout_id']}.jpg"
                batch_id = row["batch_id"]
                img_path = self.processor.image_folder / batch_id / img_name
                
                target_class, confidence = self.predictor.predict(img_path)
                if target_class is None:
                    continue

                target_class_label = self.predictor.label_map[target_class]
                row["PredictedTargetWeed"] = True if target_class_label != "non_target" else False
                row["PredictedTargetWeed_Confidence"] = confidence
                
                if confidence is not None:
                    self.saver.save_prediction(img_path, confidence, target_class_label)
                
                self.predicted_rows.append(row)
        
        result_csv = self.save_results()
        return result_csv

    def save_results(self):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_filename = self.saver.output_folder / f"prediction_batches_{timestamp}.csv"
        pd.DataFrame(self.predicted_rows).to_csv(output_filename, index=False)
        return output_filename

def main(cfg):
    """
    Main function to execute the batch inference pipeline based on configuration settings in conf/config.yaml. Uses hydra

    Args:
        cfg (dict): Configuration dictionary containing pipeline settings.
    """
    start_time = datetime.now()

    process_config = cfg['predict']
    pipeline = BatchInferencePipeline(
        image_folder=process_config['image_folder'],
        batch_prefix=process_config['batch_prefix'],
        model_path=process_config['model_path'],
        label_map=process_config['labels'],
        output_folder=process_config['output_folder'],
        date_range=(process_config['start_date'], process_config['end_date']),
        parallel_processing=process_config['parallel_inference'],
        batch_size=process_config['inference_batch_size'],
        sample_size=int(process_config['sample_folder_count'])
    )
    results = pipeline.run()
    log.info(f"Processing complete. Time spent: {datetime.now() - start_time}")
    log.info(f"Labels generated and saved to {results}")