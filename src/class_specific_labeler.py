from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import re
import json
class ImageDataLoader:
    """
    Loads and filters image data from CSV files based on configuration settings.
    """

    def __init__(self, config):
        self.config = config
        self.validated = config['filter']['validated']
        self.storage = config['filter']['storage']
        self.image_folder = Path(config['paths']['image_folder'], self.storage, "semifield-cutouts")
        self.batch_prefix = config['filter']['batch_prefix']
        self.start_date = pd.to_datetime(config['filter']['start_date'])
        self.end_date = pd.to_datetime(config['filter']['end_date'])
        self.sample_size = config['filter']['sample_size']
        self.output_folder = Path(config['paths']['output_folder'])
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.plant_classes = config['filter']['plant_classes']
        self.season_keyword = config['filter']['season_keyword']
        self.df = None

    def get_valid_batches(self, directory):
        """Identify valid batch directories based on naming pattern."""
        pattern = re.compile(r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$')
        return [folder for folder in directory.iterdir() if folder.is_dir() and pattern.match(folder.name)]
    
    def load_batches(self):
        """This method performs the following steps:
        1. Retrieves valid batches from the specified image folder.
        2. Filters batches based on the batch prefix.
        3. Filters batches within the specified date range.
        4. Samples a subset of the filtered batches based on the sample size.
        5. Reads CSV files from the sampled batches and concatenates them into a single DataFrame.
        6. Optionally filters the DataFrame to include only validated images.
        7. Converts the 'category_common_name' column to lowercase.
        8. Filters the DataFrame to include only rows where 'category_common_name' contains any of the specified plant classes.
        9. Filters the DataFrame to include only rows where 'season' contains the specified season keyword.
        Raises:
            ValueError: If no batches are found for the specified date range.
            ValueError: If no validated images are found in the specified date range.
        
        Loads batches based on the configured date range and filters data by plant classes and season.
        """
        batches = self.get_valid_batches(self.image_folder)
        batches = [batch for batch in batches if self.batch_prefix in batch.name]
        filtered_batches = [
            batch for batch in batches 
            if self.start_date <= pd.to_datetime(batch.name.split("_")[1]) <= self.end_date
        ]
        if not filtered_batches:
            raise ValueError("No batches found for the specified date range.")
        
        batch_sample = random.sample(filtered_batches, min(len(filtered_batches), self.sample_size))
        
        csv_files = [list(batch.glob("*.csv"))[0] for batch in batch_sample]
        self.df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True)
        
        if self.validated:
            self.df = self.df[self.df["validated"] == True]
        
        if self.df.empty:
            raise ValueError("No validated images found in the specified date range.")
        
        self.df["category_common_name"] = self.df["category_common_name"].str.lower()
        # self.df = self.df[self.df["category_common_name"].isin(self.plant_classes)]
        self.df = self.df[self.df["category_common_name"].apply(lambda x: any(plant_class in x for plant_class in self.plant_classes))]
        self.df = self.df[self.df["season"].str.contains(self.season_keyword, case=False, na=False)]

    def filter_existing_labels(self):
        """
        Filters out images that have already been labeled.
        """
        output_csvs = list(self.output_folder.glob("image_classes.csv"))
        if output_csvs:
            labeled_df = pd.concat([pd.read_csv(csv) for csv in output_csvs], ignore_index=True)
            self.df = self.df[~self.df["cutout_id"].isin(labeled_df["cutout_id"])]

    def get_data(self):
        """Returns the filtered DataFrame."""
        return self.df


class Sampler:
    """
    Handles stratified sampling of images by custom area categories.
    """

    def __init__(self, df, area_categories, n_samples_per_category):
        self.df = df
        self.area_categories = area_categories
        self.n_samples_per_category = n_samples_per_category
        self.selected_classes = area_categories.keys()

    def stratified_sample(self):
        """
        Performs stratified sampling across custom area categories.
        """
        # Define a custom category based on area ranges
        def categorize_area(area):
            for category, (min_area, max_area) in self.area_categories.items():
                if min_area <= area < max_area:
                    return category
            return None

        # Apply the categorization to the DataFrame
        self.df["area_category"] = self.df["cutout_props_bbox_area_cm2"].apply(categorize_area)
        self.df = self.df[self.df["area_category"].notna()]  # Filter out rows not falling into any category
        
        final_df = self.df.groupby(["category_common_name", "area_category"], group_keys=False).apply(
            lambda x: x.sample(min(len(x), self.n_samples_per_category))
        )
        final_df_sorted = final_df.sort_values(by=["cutout_props_bbox_area_cm2"])
        # Perform stratified sampling
        return  final_df_sorted 


class ImageViewer:
    """
    Displays images with labels and allows user to classify them.
    """

    def __init__(self,cfg, df, class_labels):
        self.cfg = cfg
        self.class_names = "_".join([x.replace(" ", "_") for x in cfg['filter']['plant_classes']])
        # self.image_folder = Path(cfg.paths.image_folder, self.storage, "semifield-cutouts")
        self.lts_dir = cfg.paths.image_folder
        self.output_folder = Path(cfg.paths.output_folder)
        self.results = self.load_or_create_csv(self.output_folder / f"image_classes.csv", df.columns)
        self.df = df[~df["cutout_id"].isin(self.results["cutout_id"])]
        self.class_labels = class_labels
        self.window_name = "Image Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        self.developed_image_cache = {}
        self.metadata_cache = {}

    @staticmethod
    def load_or_create_csv(file_path, columns):
        """Loads or creates a CSV for storing classification results."""
        return pd.read_csv(file_path) if file_path.exists() else pd.DataFrame(columns=[*columns, "non_target_weed_class"])

    @staticmethod
    def get_dynamic_font_scale(image_height, image_width):
        """Calculates a dynamic font scale based on image dimensions."""
        min_dim = min(image_height, image_width)
        return 0.3 if min_dim < 100 else 0.5 if min_dim < 500 else 1.5

    def read_json(self, developed_metadata_path):
        """Reads the JSON file and returns the data."""
        with open(developed_metadata_path) as f:
            return json.load(f)
        
    def display_images(self, show_developed=False):
        """
        Displays images, allowing user to classify them using keyboard inputs.

        Args:
            show_developed (bool): If True, displays the developed image alongside the cutout.
        """
        key_mapping = {ord(k): v for k, v in self.class_labels.items()}
        key_text_lines = [" | ".join([f"{chr(k)}: {v}" for k, v in key_mapping.items()][i:i+4]) for i in range(0, len(key_mapping), 4)]
        key_text = "\n".join(key_text_lines)
        # print all the possible key options
        print("\nPress 'b' to go back to the previous image.")
        print("Press 'q' to quit the program.")
        print("Press 'p' to toggle displaying the developed image.\n")
        # print the possible key mappings
        for key, value in key_mapping.items():
            print(f"\t'{chr(key)}' : {value}")
        print("\n")
        viewed_stack = []  # Stack to store previously viewed rows
        current_index = 0

        with tqdm(total=len(self.df), desc="Labeling Images", unit="image") as progress_bar:
            while current_index < len(self.df):
                row = self.df.iloc[current_index]
                image_dir = Path(self.lts_dir, "GROW_DATA", "semifield-cutouts", row["batch_id"])
                developed_image_dir = Path(self.lts_dir, "GROW_DATA", "semifield-developed-images", row["batch_id"], "images")
                developed_metadata_dir = Path(self.lts_dir, "GROW_DATA", "semifield-developed-images", row["batch_id"], "metadata")

                if not image_dir.exists():
                    image_dir = Path(self.lts_dir, "longterm_images", "semifield-cutouts", row["batch_id"])
                if not developed_image_dir.exists():
                    developed_image_dir = Path(self.lts_dir, "longterm_images", "semifield-developed-images", row["batch_id"], "images")
                    developed_metadata_dir = Path(self.lts_dir, "longterm_images", "semifield-developed-images", row["batch_id"], "metadata")

                image_path = Path(image_dir, f"{row['cutout_id']}.jpg")
                developed_image_path = Path(developed_image_dir, f"{row['image_id']}.jpg")
                developed_metadata_path = Path(developed_metadata_dir, f"{row['image_id']}.json")

                cutout_image = cv2.imread(str(image_path))
                if cutout_image is None:
                    current_index += 1
                    continue

                cutout_image = cv2.resize(cutout_image, (300, 300))
                cropped_image = None
                display_with_developed = show_developed  # Use the default setting initially

                while True:
                    # Dynamically create combined image based on the current setting
                    if display_with_developed:
                        if row["image_id"] in self.metadata_cache:
                            metadata = self.metadata_cache[row["image_id"]]
                        else:
                            metadata = self.read_json(developed_metadata_path)
                            self.metadata_cache[row["image_id"]] = metadata

                        if row["image_id"] in self.developed_image_cache:
                            developed_image = self.developed_image_cache[row["image_id"]]
                        else:
                            developed_image = cv2.imread(str(developed_image_path))
                            if developed_image is None:
                                current_index += 1
                                break
                            self.developed_image_cache[row["image_id"]] = developed_image

                        bbox = next((ann["bbox_xywh"] for ann in metadata["annotations"] if ann["cutout_id"] == row["cutout_id"]), None)
                        if bbox:
                            original_width, original_height = metadata["exif_meta"]["ImageWidth"], metadata["exif_meta"]["ImageLength"]

                            x, y, w, h = bbox
                            padding = 1000  # Pixels of context around the bounding box

                            # Add padding while ensuring we stay within image bounds
                            x1 = max(0, int(x - padding))
                            y1 = max(0, int(y - padding))
                            x2 = min(original_width, int(x + w + padding))
                            y2 = min(original_height, int(y + h + padding))
                            # Crop the developed image
                            cropped_image = developed_image[y1:y2, x1:x2]

                            # Resize the cropped image for display
                            cropped_image = cv2.resize(cropped_image, (400, 400))
                            # Draw bounding box on the cropped image
                            box_x = int((x - x1) * (400 / (x2 - x1)))
                            box_y = int((y - y1) * (400 / (y2 - y1)))
                            box_w = int(w * (400 / (x2 - x1)))
                            box_h = int(h * (400 / (y2 - y1)))
                            cv2.rectangle(cropped_image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
                        else:
                            cropped_image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Blank image if bbox is missing

                        # Combine cutout and developed images
                        combined_height = max(cutout_image.shape[0], cropped_image.shape[0])
                        combined_width = cutout_image.shape[1] + cropped_image.shape[1]
                        combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
                        combined_image[:cutout_image.shape[0], :cutout_image.shape[1]] = cutout_image
                        combined_image[:cropped_image.shape[0], cutout_image.shape[1]:] = cropped_image
                    else:
                        combined_image = cutout_image

                    font_scale = self.get_dynamic_font_scale(300, 300)
                    cv2.putText(combined_image, row["category_common_name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

                    cv2.imshow(self.window_name, combined_image)
                    key = cv2.waitKey(0)

                    if key in key_mapping:
                        self.results = self.results[self.results["cutout_id"] != row["cutout_id"]]
                        row = self.df.iloc[current_index].copy()
                        row["non_target_weed_class"] = key_mapping[key]
                        if self.results.empty:
                            self.results = pd.DataFrame([row])
                        else:
                            self.results = pd.concat([self.results, pd.DataFrame([row])], ignore_index=True)

                        self.save_results()
                        viewed_stack.append(current_index)
                        current_index += 1
                        progress_bar.update(1)
                        break

                    elif key == ord('b'):
                        if viewed_stack:
                            progress_bar.n -= 1
                            progress_bar.refresh()
                            current_index = viewed_stack.pop()
                            break
                        else:
                            print("No previous images to go back to.")
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('p'):
                        # Toggle displaying the developed image temporarily
                        display_with_developed = not display_with_developed

    def save_results(self):
        """Saves the classification results to a CSV file."""
        self.results.to_csv(self.output_folder / f"image_classes.csv", index=False)


def main(cfg):
    data_loader = ImageDataLoader(cfg)
    data_loader.load_batches()
    data_loader.filter_existing_labels()

    df = data_loader.get_data()
    if df.empty:
        print("No images to process.")
        return

    print(f"Number of images to label before sampling: {len(df)}")
    # Pass custom area categories for sampling
    area_categories = {
        key: tuple(value) for key, value in cfg['filter']['area_categories'].items()
    }
    sampler = Sampler(df, area_categories, cfg['filter']['n_samples_per_category'])
    sampled_df = sampler.stratified_sample()
    print(f"Number of images to label after sampling: {len(sampled_df)}")

    viewer = ImageViewer(cfg, sampled_df, cfg['filter']['class_labels'])
    viewer.display_images(show_developed=cfg.filter.show_developed)
