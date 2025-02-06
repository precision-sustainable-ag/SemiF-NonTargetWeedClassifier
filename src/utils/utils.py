from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_and_process_csvs(data_dir):
    """Load and process CSV files to create a combined DataFrame."""
    csvs = data_dir.rglob("*.csv")
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['cutout_id'])  # Remove repeated cutout_ids
    final_df['non_target_weed_class'] = final_df['non_target_weed_class'].replace('grass', 'rye')
    return final_df


def stratified_split(df, stratify_col, test_size=0.2, random_state=42):
    """Perform stratified train-validation split."""
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[stratify_col], 
        random_state=random_state
    )
    return train_df, val_df


def move_cutouts(row, cutout_dir, output_dir):
    """Move cutout images to their respective output folders."""
    common_name = row['non_target_weed_class']
    if common_name == "grass":
        common_name = "rye"
    cutout_id = row['cutout_id']
    
    cutout_path = Path(cutout_dir, f"{cutout_id}.jpg")
    if not cutout_path.exists():
        return  # Skip if the cutout file doesn't exist

    cname_output_dir = Path(output_dir, common_name)
    cname_output_dir.mkdir(parents=True, exist_ok=True)
    dst_path = Path(cname_output_dir, f"{cutout_id}.jpg")
    if not dst_path.exists():
        shutil.copy(cutout_path, cname_output_dir)

def find_cutout_dir(batch, lts_dir, lts_dir_names):
    """
    Search for the batch directory in the specified LTS directories.
    
    Parameters:
    - batch: The batch name.
    - lts_dir: The root LTS directory.
    - lts_dir_names: List of possible subdirectories in LTS.

    Returns:
    - The Path to the cutout directory if found, otherwise None.
    """
    for lts_dir_name in lts_dir_names:
        cutout_dir = Path(lts_dir, lts_dir_name, "semifield-cutouts", batch)
        if cutout_dir.exists():
            return cutout_dir
    return None

def process_and_move_images(lts_dir, lts_dir_names, train_df, val_df, output_base_dir):
    """Process and move images into train and validation folders."""
    non_existing_batches = []
    for split, df in zip(["train", "val"], [train_df, val_df]):
        output_dir = Path(output_base_dir, split)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} split"):
            batch = row['batch_id']
            cutout_dir = find_cutout_dir(batch, lts_dir, lts_dir_names)
            if not cutout_dir:
                if batch not in non_existing_batches:
                    print(f"Batch {batch} not found in any LTS directory.")
                    non_existing_batches.append(batch)
                continue
            move_cutouts(row, cutout_dir, output_dir)

if __name__ == "__main__":
    data_dir = Path("results")
    lts_dir = Path("/mnt/research-projects/s/screberg/")
    lts_dir_names = ["GROW_DATA", "longterm_images"]
    output_base_dir = Path("multi_class_data")

    # Step 1: Load and process the CSVs
    # result = load_and_process_csvs(data_dir)
    # result = result[result['non_target_weed_class'] == 'rye']
    # result.to_csv("temp_rye.csv", index=False)
    
    df = pd.read_csv("results/image_classes.csv")
    # df = df[df["non_target_weed_class"] != "horseweed"]

    # Step 2: Perform a stratified train-validation split
    train_df, val_df = stratified_split(df, stratify_col='non_target_weed_class', test_size=0.12, random_state=42)

    print("\nTrain split:")
    print(f"Total: {len(train_df)}")
    print(train_df.groupby(['non_target_weed_class']).size())

    print("\nValidation split:")
    print(f"Total: {len(val_df)}")
    print(val_df.groupby(['non_target_weed_class']).size())

    # Step 3: Move images into train and validation folders
    process_and_move_images(lts_dir, lts_dir_names, train_df, val_df, output_base_dir)