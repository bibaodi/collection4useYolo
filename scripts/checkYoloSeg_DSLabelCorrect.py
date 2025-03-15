import os
import shutil
import logging

# Configure logging
logging.basicConfig(filename='segLabel_check.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def is_valid_label_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            Nof = len(lines)
            if 1!=Nof:
                return False
            for line in lines:
                parts = line.split()
                if len(parts) < (3*2+1):#3points(x,y) + classIndex=7
                    return False
                class_id =int(parts[0])
                if not (0 <= class_id < 100):
                    logging.error(f"class_id Err:{class_id}")
                    return False
                coordinates = map(float, parts[1:])
                for icord in coordinates:
                    if icord <0 or icord >1:
                        return False
        return True
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return False

def move_to_error_folder(file_path, error_folder):
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    shutil.move(file_path, os.path.join(error_folder, os.path.basename(file_path)))
    logging.info(f"Moved invalid label file to error folder: {file_path}")

def check_labels_in_folder(folder_path, error_folder):
    msg=f"process: {folder_path}"
    print(msg)
    filenames = os.listdir(folder_path)
    print(f"filenames len={len(filenames)}")
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            if not is_valid_label_file(file_path):
                move_to_error_folder(file_path, error_folder)

if __name__ == "__main__":
    labels_folder = '/home/eton/00-src/yolo-ultralytics-250101/datasets/thyNoduBoMSegV01/labels'  # Path to the folder containing label files
    error_folder = '/home/eton/00-src/yolo-ultralytics-250101/datasets/thyNoduBoMSegV01/label-error'  # Path to the folder where invalid labels will be moved


    subfolders=['train', 'val']
    for ifolder in subfolders:
        check_labels_in_folder(labels_folder+'/'+ifolder, error_folder)
    logging.info("Label validation completed.")

