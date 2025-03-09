import os
import shutil
import logging

# Configure logging
logging.basicConfig(filename='image_label_check.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def find_image_files(folder_path, image_extensions):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                image_files.append(os.path.join(root, file))
    return image_files

def move_to_error_folder(file_path, error_folder):
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    shutil.move(file_path, os.path.join(error_folder, os.path.basename(file_path)))
    print(f"Moved image file to error folder: {file_path}")
    logging.info(f"Moved image file to error folder: {file_path}")

def check_labels(image_files, labels_folder, error_folder):
    for image_path in image_files:
        # Construct the expected label file path
        relative_path = os.path.relpath(image_path, start='images')
        label_path = os.path.join(labels_folder, relative_path.replace('.jpg', '.txt').replace('.png', '.txt'))

        if not os.path.exists(label_path):
            move_to_error_folder(image_path, error_folder)

if __name__ == "__main__":
    dataset_folder = 'path/to/dataset'  # Path to the dataset folder
    dataset_folder = '/home/eton/00-src/yolo-ultralytics-250101/datasets/301pacsDataInLbmfmtRangeY22-24.ThyNoduOnlyV1'  # Path to the dataset folder
    images_folder = os.path.join(dataset_folder, 'images')
    labels_folder = os.path.join(dataset_folder, 'labels')
    error_folder = os.path.join(dataset_folder, 'error-imgs')  # Path to the error folder

    image_extensions = ['.jpg', '.png', '.jpeg']  # Add more extensions if needed

    # Find all image files
    image_files = find_image_files(images_folder, image_extensions)

    # Check for corresponding label files
    check_labels(image_files, labels_folder, error_folder)

    logging.info("Image and label validation completed.")

