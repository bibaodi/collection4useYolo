from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os

# Constants
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def process_results(results, output_dir="/tmp"):
    """Process model results and save output images"""
    for result in results:
        img = np.copy(result.orig_img)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        
        # Process mask and create isolated image
        contour = result.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        
        # Generate output filenames based on input path
        input_path = result.path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_base = os.path.join(output_dir, base_name)
        
        cv2.imwrite(f"{output_base}-isolated.png", np.dstack([img, b_mask]))
        cv2.drawContours(img, [contour], -1, (55, 255, 25), 
                       thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{output_base}-contour.png", img)

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
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    image_paths = get_image_paths(args)
    run_processing_pipeline(args.model_file, image_paths, args.output_dir)

def run_processing_pipeline(model_file, image_paths, output_dir):
    """Execute full processing pipeline with batching"""
    model = YOLO(model_file)
    batch_size = 32
    
    for i in range(0, len(image_paths), batch_size):
        results = model.predict(image_paths[i:i + batch_size])
        process_results(results, output_dir)

if __name__ == "__main__":
    main()