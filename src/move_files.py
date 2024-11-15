"""
Function to move files from longterm storage locations, and split it into train and validation datasets
"""
from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

log = logging.getLogger(__name__)

def determine_class(row, reversed_mapping):
    """
    
    reversed_mapping is common name -> classname mapping (eg: "crimson clover": "clover")
    mapping in config.yaml is classname -> array of common name mapping (eg: "clover": ["crimson clover", "red clover"])
    reversed_mapping reduces the computations needed at each step but mapping in config.yaml is easier to read
    """
    if row['common_name'].lower() in reversed_mapping.keys() and row['TargetWeed']:
        return reversed_mapping[row['common_name'].lower()]
    elif row['TargetWeed']:
        return row['common_name'].lower()
    else:
        return "non_target"
    
# Helper function to copy a single image
def copy_single_image(row, subset, dest, lts_locations):
    """Copy a single image based on the DataFrame row."""

    image_name = row["cutout_id"] + ".jpg"
    i = 0
    source = Path(os.path.join(lts_locations[i], row["batch_id"], image_name))
    while not source.exists():
        if i< len(lts_locations):
            source = Path(os.path.join(lts_locations[i], row["batch_id"], image_name))
            i+= 1
        else:
            break

    if source.exists():
        # targetweed = row["TargetWeed"]
        targetweed = row["class"]
        # Set the target folder based on class and subset (train/val)
        # target_folder = dest / subset / ("target_grass" if targetweed else "non_target")
        target_folder = Path(os.path.join(dest, subset, targetweed))
        target_folder.mkdir(exist_ok=True, parents=True)

        target = target_folder / image_name
        shutil.copy2(Path(source), target)
    else:
        print(f"Image {source} not found")

# Function to copy images using a thread pool
def copy_images_parallel(df, subset, dest, lts_locations, max_workers):
    """Copy images in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_single_image, row, subset, dest, lts_locations) for _, row in df.iterrows()]
        for future in as_completed(futures):
            try:
                future.result()  # Raise exceptions if any occurred during execution
            except Exception as e:
                print(f"Error copying image: {e}")

def main(cfg):
    """
     Main function that uses the hydra config and moves files for training
    """
    task_config = cfg['move_files']
    
    labels_folder = task_config['labels_folder']
    state_prefix = task_config['batch_prefix']
    image_folders = task_config['longterm_storage_locations']
    output_folder = task_config['output_folder']
    common_name_mapping = task_config['common_name_grouping']
    reversed_name_mapping = {}
    for k, v in common_name_mapping.items():
        for common_name in v:
            reversed_name_mapping[common_name] = k

    output_csvs = [x for x in Path(labels_folder).rglob("*.csv")]
    df = pd.concat([pd.read_csv(csv) for csv in output_csvs], ignore_index=True)
    log.info(f"Starting off with {len(df)} labels across all locations")

    df = df[df["batch_id"].str.contains(state_prefix)]
    df = df.drop_duplicates(subset=["cutout_id"])
    df = df[df["common_name"] != "unknown"]
    log.info(f"Total number of images: {len(df)}")

    df["class"] = df.apply(determine_class, args=(reversed_name_mapping, ), axis=1)

    # Perform train/val split (80% train, 20% val)
    train_df, val_df = train_test_split(df, test_size=task_config['test_size'], stratify=df["class"], random_state=42)
    # train_df = val_df = df

    log.info(f"Number of training images: {len(train_df)}")
    log.info(f"Number of validation images: {len(val_df)}")
    # print(f"Number of target weed train images: {train_df[train_df['class'] == True].shape[0]}")
    log.info(f"Number of target weed grass train images: {train_df[train_df['class'] == 'grass'].shape[0]}")
    log.info(f"Number of target weed broadleaf train images: {train_df[train_df['class'] == 'broadleaf'].shape[0]}")
    log.info(f"Number of target weed hairy vetch train images: {train_df[train_df['class'] == 'hairy vetch'].shape[0]}")
    log.info(f"Number of non-target weed train images: {train_df[train_df['class'] == 'non_target'].shape[0]}")
    log.info(f"Number of target grass weed val images: {val_df[val_df['class'] == 'grass'].shape[0]}")
    log.info(f"Number of target broadleaf weed val images: {val_df[val_df['class'] == 'broadleaf'].shape[0]}")
    log.info(f"Number of target hairy vetch weed val images: {val_df[val_df['class'] == 'hairy vetch'].shape[0]}")
    log.info(f"Number of non-target weed val images: {val_df[val_df['class'] == 'non_target'].shape[0]}")
    
    copy_images_parallel(train_df, "train", output_folder, image_folders, task_config['max_workers'])
    copy_images_parallel(val_df, "val", output_folder, image_folders, task_config['max_workers'])
