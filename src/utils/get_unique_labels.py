from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm

class BatchProcessor:
    def __init__(self, base_dir, lts_dir_names):
        self.base_dir = base_dir
        self.lts_dir_names = lts_dir_names

    def get_valid_batches(self, directory):
        """Identify valid batch directories based on naming pattern."""
        pattern = re.compile(r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$')
        return [folder for folder in directory.iterdir() if folder.is_dir() and pattern.match(folder.name)]

    def get_validated_directory(self, lts_dir_name):
        lts_dir = Path(self.base_dir, lts_dir_name, "semifield-cutouts")
        valid_batches = self.get_valid_batches(lts_dir)

        dfs = []
        for valid_batch in tqdm(valid_batches, desc=f"Processing batches in {lts_dir_name}", leave=False):
            csv = valid_batch / f"{valid_batch.name}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv, low_memory=False)
            df["lts_id"] = lts_dir_name
            df['state'] = df['batch_id'].apply(lambda x: x.split('_')[0])
            dfs.append(df)

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        # df["lts_id"] = lts_dir_name

        # # Extract state from batch_id
        # df['state'] = df['batch_id'].apply(lambda x: x.split('_')[0])

        # Add batch count for each group
        df['batch_id'] = df['batch_id'].astype(str)  # Ensure batch_id is string for unique count
        batch_counts = df.groupby(['state', 'lts_id', 'season','validated',])['batch_id'].nunique().reset_index(name='batch_count')

        # Get unique common names for each group and order them alphabetically
        unique_common_names = (
            df.groupby(['state','lts_id', 'season','validated',])['category_common_name']
            .apply(lambda names: sorted(names.unique()))  # Sort the unique names alphabetically
            .reset_index(name='unique_common_names')
        )

        # Merge unique common names with batch counts
        merged_df = pd.merge(unique_common_names, batch_counts, on=['state', 'lts_id', 'season','validated'])
        return merged_df, dfs

    def process_all_validated_directories(self):
        all_dfs = []
        new_df = []
        for lts_dir_name in tqdm(self.lts_dir_names, desc="Processing LTS directories"):
            merged_df, dfs = self.get_validated_directory(lts_dir_name)
            if merged_df is not None:
                new_df.append(merged_df)
                all_dfs.extend(dfs)

        if new_df:
            final_df = pd.concat(new_df, ignore_index=True)
            return final_df, all_dfs
        else:
            return pd.DataFrame(), all_dfs

    def save_results(self, final_df, output_file):
        final_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    base_dir = "/home/mkutuga/SemiF-CleanData/data/cleaned"
    lts_dir_names = ["GROW_DATA", "longterm_images"]
    output_file = "conf/unique_labels.csv"
    output_master_file = "conf/master.csv"

    processor = BatchProcessor(base_dir, lts_dir_names)
    final_df, dfs = processor.process_all_validated_directories()
    final_df['general_season'] = final_df['season'].apply(lambda x: 'cash' if 'cash' in x.lower() else ('covers' if 'cover' in x.lower() else 'weeds'))
    df = pd.concat(dfs, ignore_index=True)
    df['general_season'] = df['season'].apply(lambda x: 'cash' if 'cash' in x.lower() else ('covers' if 'cover' in x.lower() else 'weeds'))
    print(df.shape)
    processor.save_results(final_df, output_file)
    processor.save_results(df, output_master_file)
    
    df = pd.read_csv(output_file)
    
    # pd.set_option('display.max_colwidth', None)
    
    # Filter for rows where 'season' contains "covers" and 'state' is "MD"
    covers_md = df[(df['season'].str.contains("covers", case=False, na=False)) & (df['state'] == "MD")]
    total_batches = covers_md['batch_count'].sum()
    common_names = set()
    for names in covers_md['unique_common_names']:
        # Convert string representation of the list back to Python list
        common_names.update(eval(names))
    # Display results
    print(f"All unique common names for 'covers' in season and state 'MD':")
    print(sorted(common_names))
    print(f"Total batches: {total_batches}")