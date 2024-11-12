import os
import logging
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime  # Import for timestamp
import numpy as np
from tqdm import tqdm
import random

log = logging.getLogger(__name__)

def df_filter(df, image_folder):
    # Get list of images
    images = [Path(img).name for img in os.listdir(image_folder) if img.endswith(".jpg")]
    df = df[df["Name"].isin(images)]
    return df

def stratified_sample(df: pd.DataFrame, n_samples_per_bin=5, max_bins=10):
    """
    Generate an even sample across the smallest area bins for each species.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data to sample from.
        n_samples_per_bin (int): Number of samples to take per bin for each species.
        max_bins (int): Number of the smallest bins to retain for sampling.
    """
    # Define logarithmic bins for the area
    bins = np.logspace(np.log10(df["area"].min() + 1), np.log10(df["area"].max() + 1), num=10)
    df["area_bin"] = pd.cut(df["area"], bins=bins, labels=range(len(bins) - 1), duplicates="drop")
    
    # Select only the smallest bins
    smallest_bins = range(max_bins)  # Select the first few (smallest) bins
    df = df[df["area_bin"].isin(smallest_bins)]
    
    # Perform stratified sampling across the filtered bins
    sampled_df = df.groupby(["common_name", "area_bin"], group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_samples_per_bin))
    )
    return sampled_df

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

def filter_by_season(df, keyword):
    """Filter rows where 'Seasons' column contains the word 'cover'."""
    return df[df["season"].str.contains(keyword, case=False, na=False)]

def main(cfg):
    """
    Main function to label images based on configuration settings in conf/config.yaml. Uses hydra

    Args:
        cfg (dict): Configuration dictionary containing pipeline settings.
    """
    task_config = cfg['label_images']
    image_folder = Path(task_config['image_folder'])
    season = task_config['season']
    batch_prefix = task_config['batch_prefix']
    sample_size = task_config['sample_folder_count']
    data_folder = Path(os.path.join(task_config['parent_output_folder'], f"{batch_prefix}_{season}"))
    start_date = task_config['start_date']
    end_date = task_config['end_date']
    
    
    log.info("Loading batches")
    batches = [x for x in image_folder.glob('*') if batch_prefix in x.stem and (pd.to_datetime(start_date) <= pd.to_datetime(x.stem.split('_')[1]) <= pd.to_datetime(end_date))]
    if sample_size >= len(batches):
        batch_sample = batches
    else:
        batch_sample = random.sample(batches, sample_size)
    
    log.info(f"{len(batch_sample)} batches used from {len(batches)} batches available")
    
    # should go here since it's possible this is the first usage
    data_folder.mkdir(exist_ok=True, parents=True)
    csvs = [x for x in data_folder.rglob("*.csv")]
    if csvs:
        # this codeblock gets all images from the first csv and keeps the cutouts if they are present in long term storage
        cutout_ids = [x.stem for x in image_folder.glob("*.jpg")]
        df = pd.read_csv([x for x in data_folder.glob("*.csv")][0])
        df = df[df["cutout_id"].isin(cutout_ids)]
        
        # this removes cutouts from the dataframe if they are present in data_folder
        labeled_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)
        df = df[~df["cutout_id"].isin(labeled_df["cutout_id"])]
    
    log.info(f"Size of df before filtering: {len(df)}")
    exit()
    df = filter_by_season(df, season)
    log.info(f"Size of df after filtering: {len(df)}")

    df = stratified_sample(df, n_samples_per_bin=20, max_bins=6)

    if len(df) == 0:
        print("No images to process")
        exit()
    
    log.info(f"Processing {len(df)} images")
    image_viewer(df, image_folder, data_folder)



# if __name__ == "__main__":
#     # Example usage
#     batch_prefix = "MD"

#     image_folder = Path("/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts")
#     # image_folder = Path("/mnt/research-projects/s/screberg/GROW_DATA/semifield-cutouts")
 
#     data_folder = Path("../labels/md_covers")
#     csvs = [x for x in data_folder.rglob("*.csv")]
#     cutout_ids = [x.stem for x in image_folder.glob("*.jpg")]
#     df = pd.read_csv([x for x in data_folder.glob("*.csv")][0])
#     df = df[df["cutout_id"].isin(cutout_ids)]

    
#     print(f"Size of df before filtering: {len(df)}")
    
#     data_folder.mkdir(exist_ok=True, parents=True)

#     # Load all CSV files into a single DataFrame
#     output_csvs = [x for x in data_folder.glob("*.csv")]

        
#     if output_csvs:
#         labeled_df = pd.concat([pd.read_csv(csv) for csv in output_csvs], ignore_index=True)
#         df = df[~df["cutout_id"].isin(labeled_df["cutout_id"])]

#     df = filter_by_season(df, "cover")
#     print(f"Size of df before filtering: {len(df)}")
#     df = stratified_sample(df, n_samples_per_bin=20, max_bins=6)
#     if len(df) == 0:
#         print("No images to process")
#         exit()
    
#     print(f"Processing {len(df)} images")
#     image_viewer(df, image_folder, data_folder)