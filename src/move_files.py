from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

log = logging.getLogger(__name__)

def determine_class(row):
    if row["common_name"].lower() == "hairy vetch" and row["TargetWeed"]:
        return "hairy vetch"
    if row["TargetWeed"]:
        return "grass" if row["group"].lower() == "monocot" else "broadleaf"
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
    # if not source.exists():
    #     source = image_folder2 / row["batch_id"] / image_name

    # if not source.exists():
    #     source = image_folder3 / row["batch_id"] / image_name

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
    task_config = cfg['move_files']
    
    labels_folder = task_config['labels_folder']
    state_prefix = task_config['batch_prefix']
    image_folders = task_config['longterm_storage_locations']
    output_folder = task_config['output_folder']

    output_csvs = [x for x in Path(labels_folder).rglob("*.csv")]
    df = pd.concat([pd.read_csv(csv) for csv in output_csvs], ignore_index=True)
    log.info(f"Starting off with {len(df)} labels across all locations")

    df = df[df["batch_id"].str.contains(state_prefix)]
    df = df.drop_duplicates(subset=["cutout_id"])
    df = df[df["common_name"] != "unknown"]
    log.info(f"Total number of images: {len(df)}")

    df["class"] = df.apply(determine_class, axis=1)

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

# Paths
# image_folder1 = Path("/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts")
# image_folder2 = Path("/mnt/research-projects/s/screberg/GROW_DATA/semifield-cutouts")
# image_folder3 = Path("/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts")
# state_prefix = "MD"

# label_csvs = Path("labels/md_covers").glob("*.csv")
# dest = Path("data")  # Destination directory for copied images

# # Load CSV data
# dfs = [pd.read_csv(csv) for csv in label_csvs]
# df = pd.concat(dfs, ignore_index=True)

# # Filter rows based on batch prefix
# df = df[df["batch_id"].str.contains(state_prefix)]
# df = df.drop_duplicates(subset=["cutout_id"])
# df = df[df["common_name"] != "unknown"]

# print(f"Total number of images: {len(df)}")
# # Create a new column for class


# df["class"] = df.apply(determine_class, axis=1)
# # Perform train/val split (80% train, 20% val)
# train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["class"], random_state=42)

# print(f"Number of training images: {len(train_df)}")
# print(f"Number of validation images: {len(val_df)}")
# # print(f"Number of target weed train images: {train_df[train_df['class'] == True].shape[0]}")
# print(f"Number of target weed grass train images: {train_df[train_df['class'] == 'grass'].shape[0]}")
# print(f"Number of target weed broadleaf train images: {train_df[train_df['class'] == 'broadleaf'].shape[0]}")
# print(f"Number of target weed hairy vetch train images: {train_df[train_df['class'] == 'hairy vetch'].shape[0]}")
# print(f"Number of non-target weed train images: {train_df[train_df['class'] == 'non_target'].shape[0]}")
# print(f"Number of target grass weed val images: {val_df[val_df['class'] == 'grass'].shape[0]}")
# print(f"Number of target broadleaf weed val images: {val_df[val_df['class'] == 'broadleaf'].shape[0]}")
# print(f"Number of target hairy vetch weed val images: {val_df[val_df['class'] == 'hairy vetch'].shape[0]}")
# print(f"Number of non-target weed val images: {val_df[val_df['class'] == 'non_target'].shape[0]}")

# # print(f"Number of non-target weed train images: {train_df[train_df['class'] == False].shape[0]}")
# # print(f"Number of target weed val images: {val_df[val_df['class'] == True].shape[0]}")
# # print(f"Number of non-target weed val images: {val_df[val_df['class'] == False].shape[0]}")

# # Create 'train' and 'val' directories and copy images in parallel
# copy_images_parallel(train_df, "train", dest)
# copy_images_parallel(val_df, "val", dest)
