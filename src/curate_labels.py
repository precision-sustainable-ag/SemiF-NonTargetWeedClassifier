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

    def __init__(self, cfg):
        self.cfg = cfg
        self.validated = cfg.filter.validated
        self.non_target_classes = cfg.curate.non_target_classes

        self.output_folder = Path(cfg.paths.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_csv(self):
        csv_path = Path(self.output_folder, "image_classes.csv")
        self.df = pd.read_csv(csv_path)
        
        if self.validated:
            self.df = self.df[self.df["validated"] == True]
        
        if self.df.empty:
            raise ValueError("No validated images found in the specified date range.")
        
        self.df = self.df[self.df["non_target_weed_class"].apply(lambda x: any(plant_class in x for plant_class in self.non_target_classes))]

    def get_data(self):
        """Returns the filtered DataFrame."""
        return self.df


class ImageViewer:
    """
    Displays images with labels and allows user to classify them.
    """

    def __init__(self,cfg, df, class_labels):
        self.cfg = cfg
    
        self.lts_dir = cfg.paths.image_folder
        self.output_folder = Path(cfg.paths.output_folder)
        self.results = df
        self.class_labels = class_labels
        self.window_name = "Image Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        self.developed_image_cache = {}
        self.metadata_cache = {}

    
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
        
        with tqdm(total=len(self.results), desc="Labeling Images", unit="image") as progress_bar:
            while current_index < len(self.results):
                row = self.results.iloc[current_index]
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
                    text = f"non_t: {row['non_target_weed_class']}\ncname: {row['category_common_name']}"
                    for idx, line in enumerate(text.split("\n")):
                        cv2.putText(combined_image, line, (10, 15 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                    # cv2.putText(combined_image,text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

                    cv2.imshow(self.window_name, combined_image)
                    key = cv2.waitKey(0)

                    if key in key_mapping:
                        # Update the `non_target_weed_class` directly in the DataFrame
                        self.results.loc[self.results["cutout_id"] == row["cutout_id"], "non_target_weed_class"] = key_mapping[key]

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
    data_loader.load_csv()

    df = data_loader.get_data()
    if df.empty:
        print("No images to process.")
        return

    print(f"Loaded {len(df)} images.")
    viewer = ImageViewer(cfg, df, cfg.filter.class_labels)
    viewer.display_images(show_developed=cfg.filter.show_developed)
