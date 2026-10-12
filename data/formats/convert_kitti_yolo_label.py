import os
import random
import shutil
from PIL import Image


# Function to convert KITTI format to YOLO format for a single file
def kitti_to_yolo(input_file, output_file, image_width, image_height):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            fields = line.strip().split()

            # Read type and filter only 'Person' and 'Car'
            obj_type = fields[0]
            if obj_type not in ['Person', 'Car']:
                continue

            # Extract bounding box coordinates
            x_min = float(fields[4])
            y_min = float(fields[5])
            x_max = float(fields[6])
            y_max = float(fields[7])

            # Convert to YOLO format
            bbox_center_x = (x_min + x_max) / 2.0
            bbox_center_y = (y_min + y_max) / 2.0
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Normalize values by image dimensions
            bbox_center_x /= image_width
            bbox_center_y /= image_height
            bbox_width /= image_width
            bbox_height /= image_height

            # Map object types to YOLO class IDs
            class_id = 0 if obj_type == 'Person' else 1

            # Write to YOLO file
            outfile.write(f"{class_id} {bbox_center_x:.6f} {bbox_center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


# Function to split dataset into training and validation sets
def split_dataset(label_folder, train_ratio=0.7):
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    random.shuffle(label_files)

    split_index = int(len(label_files) * train_ratio)
    train_files = label_files[:split_index]
    val_files = label_files[split_index:]

    return train_files, val_files


# Function to process and split all files in folders
def process_and_split_folders(image_folder, label_folder, train_image_folder, val_image_folder, train_label_folder,
                              val_label_folder, train_ratio=0.7):
    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)
    if not os.path.exists(val_image_folder):
        os.makedirs(val_image_folder)
    if not os.path.exists(train_label_folder):
        os.makedirs(train_label_folder)
    if not os.path.exists(val_label_folder):
        os.makedirs(val_label_folder)

    # Split dataset
    train_files, val_files = split_dataset(label_folder, train_ratio)

    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            input_file = os.path.join(label_folder, label_file)

            # Determine output folder for labels and images
            if label_file in train_files:
                output_label_folder = train_label_folder
                output_image_folder = train_image_folder
            elif label_file in val_files:
                output_label_folder = val_label_folder
                output_image_folder = val_image_folder
            else:
                continue

            output_file = os.path.join(output_label_folder, label_file)

            # Get corresponding image dimensions and copy the image
            image_file = os.path.splitext(label_file)[0] + '.png'  # Assuming images are in PNG format
            image_path = os.path.join(image_folder, image_file)

            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_file} not found for label {label_file}")
                continue

            # Copy image file to the respective folder
            shutil.copy(image_path, output_image_folder)

            # Open image to get its dimensions
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Convert KITTI to YOLO format
            kitti_to_yolo(input_file, output_file, image_width, image_height)
            print(f"Converted {label_file} and copied image to {output_image_folder}.")


# Example usage
if __name__ == "__main__":
    # Specify folders
    image_folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_object_image_2/training/image_2"  # Path to folder containing images
    label_folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/data_object_label_2/training/label_2"  # Path to folder containing KITTI label files
    train_image_folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_format/images/train"  # Path to training image output folder
    val_image_folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_format/images/val"  # Path to validation image output folder
    train_label_folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_format/labels/train"  # Path to training label output folder
    val_label_folder = "/home/yu/zjh/SSDA-YOLO-copy/dataset3/yolo_format/labels/val"  # Path to validation label output folder

    # Process and split dataset
    process_and_split_folders(image_folder, label_folder, train_image_folder, val_image_folder, train_label_folder,
                              val_label_folder)
    print(
        f"Conversion and splitting completed. YOLO labels and images saved in {train_label_folder}, {val_label_folder} and their corresponding image folders.")

