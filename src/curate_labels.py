import os
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime  # Import for timestamp
import numpy as np
from tqdm import tqdm
import random
import logging

log = logging.getLogger(__name__)


def get_dynamic_font_scale(image_height, image_width):
    """Dynamically adjust the font scale based on image size."""
    # Use smaller font for very small images and larger font for larger images
    min_dim = min(image_height, image_width)
    
    if min_dim < 100:
        return 0.3  # Small font for very small images
    elif min_dim < 500:
        return 0.5  # Medium font for moderate-size images
    else:
        return 1.5  # Larger font for larger images


def image_viewer(df: pd.DataFrame, image_folder, output_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create DataFrame to store results
    results = pd.DataFrame(columns=[x for x in df.columns] + ["TargetWeed"])

    # Create a single OpenCV window to reuse
    window_name = "Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Keep window on top
    counter = 0
    # Main loop to process images
    for _, row in df.iterrows():
        batch_id = row["batch_id"]
        image_name = row["cutout_id"] + ".jpg"
        image_path = Path(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Resize image if necessary
        h, w = image.shape[:2]
        if h > 500 or w > 500:
            resize = (image.shape[1] // 3, image.shape[0] // 3)
            image = cv2.resize(image, resize)
        elif h < 500 or w < 500:
            resize = (image.shape[1] * 3, image.shape[0] * 3)
            image = cv2.resize(image, resize)

        # Overlay the `common_name` on the image
        label = f"{row['common_name']}"
        predicted_label = None
        
        if "PredictedTargetWeed" in row:
            predicted_label = f"{row['PredictedTargetWeed']}"
            label = row['common_name'] + "\n" + predicted_label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = get_dynamic_font_scale(h, w)
        color = (0, 255, 0)  # Green color
        thickness = max(1, int(font_scale * 2))  # Adjust thickness based on font size
        position = (10, 30)  # Top-left corner of the image

        if "PredictedTargetWeed" in row:
            for idx, line in enumerate(label.split("\n")):
                position = (10, 30 + idx * 30)
                # Add the label to the image
                cv2.putText(image, line, position, font, font_scale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

        valid_input = False

        while not valid_input:
            # Display the image with the label in the same window
            cv2.imshow(window_name, image)
            key = cv2.waitKey(0)

            # Check for valid rating input (1-9 keys correspond to 49-57 ASCII values)
            if 49 <= key <= 50:
                if key == 49:
                    target  = True
                elif key == 50:
                    target = False
                
                current_result = pd.DataFrame(row).T
                current_result["TargetWeed"] = target


                # Concatenate the current result with the main results DataFrame
                results = pd.concat([results, current_result], ignore_index=True)
                # Save results to CSV
                output_csv = os.path.join(output_folder, f"image_classes_{timestamp}.csv")
                results.to_csv(output_csv, index=False)
                print(f"Counter: {counter}, Image: {image_name}, Shape: {h,w}, TargetWeed: {target}, Prediction: {predicted_label}")
                counter += 1
                valid_input = True

            # Close the program if 'q' is pressed (optional)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return  # Exit the function early if 'q' is pressed

    # Close the window after processing all images
    cv2.destroyAllWindows()

    # Save results to CSV
    results.to_csv(output_csv, index=False)

def main(cfg):
    task_config = cfg['curate_labels']
    
    # check ../predictions_test vs predictions_test
    data_folder = task_config['data_folder']
    labels_folder = task_config['labels_folder']
    check_target_folder = task_config['check_target_class']
    min_conf = int(task_config['min_confidence']*100)
    max_conf = int(task_config['max_confidence']*100)
    
    image_folders = []
    for folder in [ x.parts[-1] for x in Path(data_folder).iterdir() if x.is_dir()]:
        folder_parts = str(folder).split('_')
        print(folder_parts)
        if (check_target_folder and len(folder_parts) == 3) or (not check_target_folder and len(folder_parts) == 4):
            if int(folder_parts[-2]) >= min_conf and int(folder_parts[-1]) <= max_conf:
                image_folders.append(str(folder))
        
    log.info(f"Curating labels for images from {image_folders}")
    image_folder = image_folders[0]
    
    cutout_ids = [x.stem for x in image_folder.glob("*.jpg")]

    # remove already processed labels
    # df still doesn't have any definition
    output_csvs = [x for x in Path(labels_folder).rglob("*.csv")]
    if output_csvs:
        labeled_df = pd.concat([pd.read_csv(csv) for csv in output_csvs], ignore_index=True)
        df = df[~df["cutout_id"].isin(labeled_df["cutout_id"])]
    
    # log.info(f"Processing {len(df)} images")
    # image_viewer(df, image_folder, output_folder)
    

if __name__ == "__main__":
    
    data_folder = Path("../predictions_test")
    
    # for all folders in data folders when they fall in range 
    image_folder = Path(data_folder, "non_target_class_35_50")
   

    csvs = [x for x in data_folder.rglob("*.csv")]
    cutout_ids = [x.stem for x in image_folder.glob("*.jpg")]
    df = pd.read_csv([x for x in data_folder.glob("*.csv")][0])
    df = df[df["cutout_id"].isin(cutout_ids)]

    # df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)
    print(f"Size of df before filtering: {len(df)}")
    
    output_folder = Path("../labels/md_covers")
    output_folder.mkdir(exist_ok=True, parents=True)

    # Load all CSV files into a single DataFrame
    output_csvs = [x for x in output_folder.glob("*.csv")]

        
    if output_csvs:
        labeled_df = pd.concat([pd.read_csv(csv) for csv in output_csvs], ignore_index=True)
        df = df[~df["cutout_id"].isin(labeled_df["cutout_id"])]

    print(f"Processing {len(df)} images")
    image_viewer(df, image_folder, output_folder)
