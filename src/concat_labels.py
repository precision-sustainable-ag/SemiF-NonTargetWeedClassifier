import os
import pandas as pd

def concatenate_csvs(folder_path, output_file, batch_prefix):
    # List to hold dataframes
    dfs = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.reset_index(drop=True, inplace=True)
    concatenated_df = concatenated_df.drop_duplicates(subset="cutout_id")
    concatenated_df = concatenated_df[concatenated_df["batch_id"].str.contains(batch_prefix)]
    concatenated_df = concatenated_df[concatenated_df["common_name"] != "unknown"]

    # Save the concatenated dataframe to a new CSV file
    concatenated_df.to_csv(output_file, index=False)

def main(cfg):
    """
    Main function that accepts hydra config
    """
    task_config = cfg['concat_labels']
    concatenate_csvs(task_config['data_folder'], task_config['output_file'], task_config['batch_prefix'])
