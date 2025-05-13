import argparse
import logging
import csv
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationPredictor as YOLOClassify
from tqdm import tqdm

class YOLOClassifier:
    """YOLO image classifier with CSV output capability"""
    
    def __init__(self, model_path: str, image_folder: str, output_csv: str, file_path_style: str = "full"):
        self.m_logger = logging.getLogger(self.__class__.__name__)
        self.m_model = YOLO(model_path,'classify', True) # Add verbose=False here to suppress model loading messages
        self.m_image_folder = Path(image_folder)
        self.m_output_csv = Path(output_csv)
        self.m_file_path_style = file_path_style
        self.m_logger.debug(f"Initialized classifier with model: {model_path}")

    def predict_images(self) -> None:
        """Process all images in the folder and write results to CSV"""
        if not self.m_image_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.m_image_folder}")

        image_paths = list(self.m_image_folder.glob("*.*"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.m_image_folder}")

        results = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                class_name, prob = self._predict_single_image(img_path)
                results.append((img_path, class_name, prob))
                self.m_logger.debug(f"Processed {img_path}: {class_name} ({prob:.2f})")
            except Exception as e:
                self.m_logger.error(f"Failed to process {img_path}: {str(e)}")

        self._write_results(results)

    def _predict_single_image(self, img_path: Path) -> Tuple[str, float]:
        """Predict single image and return top class with probability"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Invalid image file: {img_path}")
            
        results = self.m_model.predict(img, verbose=False) # Add verbose=False here to suppress model predict messages
        top1 = results[0].probs.top1
        return results[0].names[top1], results[0].probs.top1conf.item()

    def _get_formatted_path(self, img_path: Path) -> str:
        """Format image path according to specified naming strategy"""
        if self.m_file_path_style == "stem":
            return img_path.stem
        if self.m_file_path_style == "basename":
            return img_path.name
        # Default to full path
        return str(img_path.resolve())

    def _write_results(self, results: List[Tuple[Path, str, float]]) -> None:
        """Write prediction results to CSV file"""
        with open(self.m_output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "class", "probability"])
            writer.writeheader()
            for imgfilepath, cls, prob in results:
                writer.writerow({
                    "image_path": self._get_formatted_path(imgfilepath),
                    "class": cls,
                    "probability": f"{prob:.4f}"
                })
        self.m_logger.info(f"Saved results to {self.m_output_csv}")

def main():
    """Main entry point for command line execution"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="YOLO Image Classifier")
    parser.add_argument("--model", required=True, help="Path to YOLO model file")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--path-style", choices=["full", "basename", "stem"], default="stem",
                        help="Format for image paths in CSV: full path, basename, or stem")
    
    args = parser.parse_args()

    try:
        classifier = YOLOClassifier(args.model, args.images, args.output, args.path_style)
        classifier.predict_images()
    except Exception as e:
        logging.error(f"Classification failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()