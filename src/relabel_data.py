from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import re
import json

class ImageViewer:
    """
    Displays images with labels and allows user to classify them.
    """
    def __init__(self,cfg):
        self.cfg = cfg
        self.class_names = "_".join([x.replace(" ", "_") for x in cfg['filter']['plant_classes']])
        self.lts_dir = cfg.paths.image_folder
        self.specific_class = cfg.filter.plant_classes
        self.output_folder = Path("/home/mkutuga/SemiF-NonTargetWeedClassifier/results")
        self.df = self.load_or_create_csv(self.output_folder)
        self.class_labels = cfg.filter.class_labels
        self.window_name = "Image Viewer"
        self.focus_cat_cname = cfg.filter.focus_cat_cname
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

    def load_or_create_csv(self, data_dir):
        """Loads or creates a CSV for storing classification results."""
        if Path(data_dir, "relabeled_image_classes.csv").exists():
            return pd.read_csv(Path(data_dir, f"image_classes.csv"))
        
        df = pd.read_csv(Path(data_dir, f"image_classes.csv"))
        # df = df[df["non_target_weed_class"].isin(self.specific_class)]
        df = df.drop_duplicates(subset=['cutout_id'])
        return df
    
    def read_json(self, developed_metadata_path):
        """Reads the JSON file and returns the data."""
        with open(developed_metadata_path) as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def get_dynamic_font_scale(image_height, image_width):
        """Calculates a dynamic font scale based on image dimensions."""
        min_dim = min(image_height, image_width)
        return 0.3 if min_dim < 100 else 0.5 if min_dim < 500 else 1.5

    def display_images(self, show_developed=True):
        """
        Displays images, allowing user to classify them using keyboard inputs.

        Args:
            show_developed (bool): If True, displays the developed image alongside the cutout.
        """
        key_mapping = {ord(k): v for k, v in self.class_labels.items()}
        key_text_lines = [" | ".join([f"{chr(k)}: {v}" for k, v in key_mapping.items()][i:i+4]) for i in range(0, len(key_mapping), 4)]
        key_text = "\n".join(key_text_lines)
        print(f"\nKey mapping: {key_text}")\
        
        # Filter indices dynamically based on specific class
        if self.focus_cat_cname:
            filtered_indices = self.df[(self.df["non_target_weed_class"].isin(self.specific_class)) & (self.df["category_common_name"] == self.focus_cat_cname)].index.tolist()
        else:
            filtered_indices = self.df[self.df["non_target_weed_class"].isin(self.specific_class)].index.tolist()

        viewed_stack = []  # Stack to store previously viewed rows
        current_pos = 0
        # Cache for developed images
        developed_image_cache = {}
        metadata_cache = {}

        with tqdm(total=len(filtered_indices), desc="Labeling Images", unit="image") as progress_bar:
            while current_pos < len(filtered_indices):
                current_index = filtered_indices[current_pos]
            # while current_index < len(self.df):
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

                # Read cutout image
                cutout_image = cv2.imread(str(image_path))

                if cutout_image is None:
                    current_pos += 1
                    progress_bar.update(1)
                    continue

                # Resize cutout image
                cutout_image = cv2.resize(cutout_image, (300, 300))
                
                cropped_image = None  # Placeholder for the cropped developed image
                
                if show_developed:
                    # Load metadata from cache or file
                    if row["image_id"] in metadata_cache:
                        metadata = metadata_cache[row["image_id"]]
                    else:
                        metadata = self.read_json(developed_metadata_path)
                        metadata_cache[row["image_id"]] = metadata

                    # Load or retrieve the developed image from the cache
                    if row["image_id"] in developed_image_cache:
                        developed_image = developed_image_cache[row["image_id"]]
                    else:
                        developed_image = cv2.imread(str(developed_image_path))
                        # developed_image = cv2.imread(str(developed_image_path),cv2.IMREAD_GRAYSCALE)
                        if developed_image is None:
                            current_pos += 1
                            continue
                        developed_image_cache[row["image_id"]] = developed_image

                    annotations = metadata["annotations"]
                    bbox_xywh = None
                    for annotation in annotations:
                        if annotation["cutout_id"] == row["cutout_id"]:
                            bbox_xywh = annotation["bbox_xywh"]
                            break

                    if bbox_xywh:
                        # Extract bounding box region with padding
                        original_width, original_height = metadata["exif_meta"]["ImageWidth"], metadata["exif_meta"]["ImageLength"]
                        x, y, w, h = bbox_xywh
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

                # Create a side-by-side canvas
                if show_developed and cropped_image is not None:
                    combined_height = max(cutout_image.shape[0], cropped_image.shape[0])
                    combined_width = cutout_image.shape[1] + cropped_image.shape[1]
                    combined_image = np.ones((combined_height, combined_width,3), dtype=np.uint8) * 255
                    combined_image[:cutout_image.shape[0], :cutout_image.shape[1]] = cutout_image
                    combined_image[:cropped_image.shape[0], cutout_image.shape[1]:] = cropped_image
                else:
                    combined_image = cutout_image

                # Display images with category label
                font_scale = self.get_dynamic_font_scale(300, 300)
                cv2.putText(combined_image, row["category_common_name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

                while True:
                    cv2.imshow(self.window_name, combined_image)
                    key = cv2.waitKey(0)

                    if key in key_mapping:
                        self.df.loc[self.df.index[current_index], "non_target_weed_class"] = key_mapping[key]
                        self.save_results()
                        viewed_stack.append(current_pos)
                        current_pos += 1
                        progress_bar.update(1)
                        break

                    elif key == ord('b'):  # Go back
                        if viewed_stack:
                            progress_bar.n -= 1
                            progress_bar.refresh()
                            current_pos = viewed_stack.pop()
                            break
                        else:
                            print("No previous images to go back to.")
                    elif key == ord('q'):  # Quit
                        cv2.destroyAllWindows()
                        return

    def save_results(self):
        """Saves the classification results to a CSV file."""
        self.df.to_csv(self.output_folder / f"image_classes.csv", index=False)


def main(cfg):
    
    viewer = ImageViewer(cfg)
    viewer.display_images(show_developed=cfg.filter.show_developed)
