#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultrasound Nodule Segmentation Processor
=========================================

File:       useSegModelCropUSImg2NoduleOnlyRegionImg.py
Author:     Eton  (eton@company.com)
Date:       2025-03-22
Version:    1.0.0

Copyright (c) 2025 MediVision AI Labs. All rights reserved.

Description:
------------
Medical imaging processing tool for automated detection and segmentation of 
thyroid nodules in ultrasound images using YOLOv8 segmentation models. 
Provides batch processing capabilities with configurable output formats.

Features:
- Batch processing of DICOM and standard image formats
- Configurable output types (isolated masks, contour overlays, cropped regions)
- Intelligent bounding box enlargement with safety thresholds
- Multi-scale image processing preserving diagnostic quality

Usage:
------
$ python useSegModelCropUSImg2NoduleOnlyRegionImg.py -m best.pt -i input/ -o output/

License:
--------
Proprietary software. Contact author for licensing details.

Maintenance History:
--------------------
2025-03-22 (v1.0.0) - Initial release (Eton )
"""

# Standard library imports
import os
import sys
import argparse
import logging
from typing import List, Tuple, Set

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# Add type hints throughout the class
from typing import List, Tuple, Set

# Update logging configuration
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Configure handlers - file for all logs, console for errors only
file_handler = logging.FileHandler(f"{script_name}.log")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)

class ImageSegmenter:
    """Main processor for segmentation and image cropping operations"""
    
    def __init__(self, output_dir: str, enlarge_thresholds: Tuple[int, int], 
               enlarge_percent: int, save_types: List[str], min_save_size: int,
               input_root: str = None) -> None:
        if enlarge_percent < 10 or enlarge_percent > 1000:
            raise ValueError("Enlarge percentage must be between 10 and 1000")
        if min_save_size < 16:
            raise ValueError("Minimum save size must be at least 1 pixel")

        # Member variables with m_ prefix
        self.m_output_dir = output_dir
        self.m_input_root = input_root
        self.m_enlarge_thresholds = enlarge_thresholds
        self.m_enlarge_percent = enlarge_percent
        self.m_min_save_size = min_save_size
        self.m_save_types = self._parse_save_types(save_types)
        os.makedirs(self.m_output_dir, exist_ok=True)

    def _parse_save_types(self, save_types: List[str]) -> Set[str]:
        """Validate and normalize save types input"""
        if 'all' in save_types:
            return {'isolated', 'contour', 'crop', 'enlarged-crop'}
        valid_types = {'isolated', 'contour', 'crop', 'enlarged-crop'}
        userSaveTypes = {t for t in save_types if t in valid_types}
        logging.info(f"Save types: {userSaveTypes}")
        return userSaveTypes

    # Context manager methods
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _save_all_versions(self, img, contour, output_base):
        """Save selected image types based on configuration"""
        if 'isolated' in self.m_save_types:
            self._save_isolated_image(img, contour, output_base)
        if 'contour' in self.m_save_types:
            self._save_contour_overlay(img, contour, output_base)
        if 'crop' in self.m_save_types:
            self._save_basic_crop(img, contour, output_base)
        if 'enlarged-crop' in self.m_save_types:
            self._save_enlarged_crop(img, contour, output_base)

    def process_batch(self, results):
        """Process a batch of YOLO detection results"""
        for result in results:
            self._process_single_result(result)
            
    def _process_single_result(self, result):
        """Process individual detection result"""
        img = np.copy(result.orig_img)
        contour = self._get_contour(result)
        
        output_base = self._get_output_base(result.path)
        self._save_all_versions(img, contour, output_base)
        
    def _get_contour(self, result):
        """Extract and format segmentation contour"""
        contour = result.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
        if contour.size < 3:  # Minimum 3 points for a contour
            raise ValueError("Invalid contour detected")
        return contour
    
    def _save_isolated_image(self, img, contour, output_base):
        """Save image with segmentation mask"""
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        cv2.imwrite(f"{output_base}-isolated.png", np.dstack([img, mask]))
        
    def _save_contour_overlay(self, img, contour, output_base):
        """Save image with contour visualization"""
        viz_img = img.copy()
        cv2.drawContours(viz_img, [contour], -1, (55, 255, 25), 
                       thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{output_base}-contour.png", viz_img)
        
    def _save_basic_crop(self, img, contour, output_base):
        x, y, w, h = cv2.boundingRect(contour)
        if w < self.m_min_save_size or h < self.m_min_save_size:  # Fixed prefix
            return
        cv2.imwrite(f"{output_base}-crop.png", img[y:y+h, x:x+w])
        
    def _save_enlarged_crop(self, img, contour, output_base):
        x, y, w, h = cv2.boundingRect(contour)
        min_thresh, max_thresh = self.m_enlarge_thresholds  # Fixed
        
        extend_w = np.clip(int((w * self.m_enlarge_percent)/200),  # Fixed
                          min_thresh, max_thresh)
        extend_h = np.clip(int((h * self.m_enlarge_percent)/200),  # Fixed
                          min_thresh, max_thresh)
        
        # Calculate bounded coordinates
        new_x = max(0, x - extend_w)
        new_y = max(0, y - extend_h)
        new_x_end = min(img.shape[1], x + w + extend_w)
        new_y_end = min(img.shape[0], y + h + extend_h)
        
        new_w = new_x_end - new_x
        new_h = new_y_end - new_y
        if new_w < self.m_min_save_size or new_h < self.m_min_save_size:
            return
            
        cv2.imwrite(f"{output_base}-enlarged-crop.png", 
                  img[new_y:new_y_end, new_x:new_x_end])
    
    def _get_output_base(self, input_path):
        """Generate output path preserving input structure"""
        if self.m_input_root:
            rel_path = os.path.relpath(os.path.dirname(input_path), self.m_input_root)
            dest_dir = os.path.join(self.m_output_dir, rel_path)
        else:
            dest_dir = self.m_output_dir
        
        os.makedirs(dest_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(dest_dir, base_name)  # Added return statement




def run_processing_pipeline(
    model_file: str,
    image_paths: List[str],
    output_dir: str,
    enlarge_thresholds: Tuple[int, int],
    enlarge_percent: int,
    save_types: List[str],
    min_save_size: int,
    input_root: str = None  # Add new parameter
    ) -> None:
    """Execute full processing pipeline with batching"""
    try:
        logging.info(f"Starting processing pipeline for {len(image_paths)} images , output to {output_dir}")
        model = YOLO(model_file)
        if model.task != 'segment':
            raise ValueError(f"Model {model_file} is not a segmentation model")
            
        with ImageSegmenter(
            output_dir,  # Correct first parameter
            enlarge_thresholds,
            enlarge_percent,
            save_types,
            min_save_size,
            input_root=input_root
        ) as processor:
            batch_size = 32
            progress_bar = tqdm(total=len(image_paths), desc="Processing images", unit="img")
            
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                results = model.predict(batch_paths, verbose=False)
                processor.process_batch(results)
                processed = min(i + batch_size, len(image_paths))
                progress_bar.update(len(batch_paths))
                logging.info(f"Processed {processed}/{len(image_paths)} images")
                
            progress_bar.close()
                
    except Exception as e:
        logging.error(f"Error processing pipeline: {str(e)}", exc_info=True)
        exit(1)


def get_image_paths(args):
    """Handle input sources and return image paths"""
    if args.image_dir:
        return collect_images_from_dir(args.image_dir)
    return args.image_path or ["../datasets/42-minibatch/thynodu-t01.jpg"]

def collect_images_from_dir(image_dir):
    """Recursively collect images from directory"""
    image_paths = []
    for root, _, files in os.walk(image_dir):
        image_paths.extend(
            os.path.join(root, f) 
            for f in files 
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )
    return sorted(image_paths)

def parse_arguments():
    """Configure and parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Segment nodules in ultrasound images")
    parser.add_argument("-m", "--model_file", default="runs/segment/train9/weights/best.pt",
                      help="Path to trained model weights")
    parser.add_argument("-i", "--image_path", nargs='+', 
                      help="Path(s) to input image(s) for processing")
    parser.add_argument("-d", "--image_dir",
                      help="Directory containing images to process")
    parser.add_argument("-o", "--output_dir", default="/tmp",
                      help="Output directory for processed images")
    parser.add_argument("--enlarge-thresh", nargs=2, type=int, default=[32, 320],
                      metavar=("MIN", "MAX"), 
                      help="Enlargement thresholds in pixels (default: 32 320)")
    parser.add_argument("--enlarge-percent", type=int, default=100, 
                      choices=range(10, 1001), metavar="10-1000",
                      help="Enlargement percentage (default: 100)")
    parser.add_argument("--save-types", nargs='+', default=['enlarged-crop'],
                      choices=['all', 'isolated', 'contour', 'crop', 'enlarged-crop'],
                      help="Image types to save: 'all' or space-separated list (default: enlarged-crop)")
    parser.add_argument("--min-save-size", type=int, default=64,
                      help="Minimum size in pixels (both dimensions) required to save crops")
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    if not (10 <= args.enlarge_percent <= 1000):
        raise ValueError("Enlarge percentage must be between 10 and 1000")

    # Log all runtime parameters
    logging.info("Runtime Parameters:")
    logging.info(f"  Model: {args.model_file}")
    logging.info(f"  Input: {args.image_dir or args.image_path}")
    logging.info(f"  Output Directory: {args.output_dir}")
    logging.info(f"  Enlarge Thresholds: {args.enlarge_thresh}")
    logging.info(f"  Enlarge Percentage: {args.enlarge_percent}%")
    logging.info(f"  Save Types: {args.save_types}")
    logging.info(f"  Minimum Save Size: {args.min_save_size}px")
    
    image_paths = get_image_paths(args)
    
    # Pass input root directory if using image_dir
    run_processing_pipeline(args.model_file, image_paths, args.output_dir,
                          args.enlarge_thresh, args.enlarge_percent, args.save_types,
                          args.min_save_size, input_root=args.image_dir)

if __name__ == "__main__":
    main()


