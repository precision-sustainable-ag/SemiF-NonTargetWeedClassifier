from ultralytics import YOLO
from pathlib import Path
import shutil
import json

class CutoutInference:
    """Class for performing inference and saving non-target cutouts."""

    def __init__(self, model_path, data_path, output_path, label_map):
        self.model = YOLO(model_path)
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.label_map = label_map
        self.output_path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def read_json(file_path):
        """Read and return JSON data from a file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def run_inference(self):
        """Perform inference on all cutouts and save non-target ones."""
        
        images = [img for img in self.data_path.rglob("*.jpg")]
        print(f"Found {len(images)} images to process.")
        images = images[:10000]
        results = self.model(images, imgsz=128, stream=True)
        for result in results:

            img = Path(result.path)
        

            batch_id = img.parent.name
            target_class = self.label_map[result.probs.top1]

            if target_class == "non_target":
                self.save_cutout(img, batch_id)

    def save_cutout(self, img, batch_id):
        """Save non-target cutouts to the output folder."""
        croput_str = str(img)
        croput_metadata_file = croput_str.replace(".jpg", ".json")
        mask_file = croput_str.replace(".jpg", "_mask.png")
        cutout_file = croput_str.replace(".jpg", ".png")

        dest = self.output_path / batch_id
        dest.mkdir(exist_ok=True, parents=True)

        shutil.move(img, dest)
        shutil.move(croput_metadata_file, dest)
        shutil.move(mask_file, dest)
        shutil.move(cutout_file, dest)


    def run(self):
        """Run the inference and save process."""
        self.run_inference()
        print("Inference completed and non-target cutouts saved.")

if __name__ == "__main__":
    MODEL_PATH = "runs/classify/NC_covers_grasses_binary/batch32_imgsz128_5468_n/weights/best.pt"
    DATA_PATH = "/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts/"
    OUTPUT_PATH = "non_target_weeds"

    LABEL_MAP = {
        0: "non_target",
        1: "grass_weed",
    }

    inferencer = CutoutInference(MODEL_PATH, DATA_PATH, OUTPUT_PATH, LABEL_MAP)
    inferencer.run()
