import os
import shutil
import yaml
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


# Define a function to set up the folder structure for training, validation, and testing
def initialize_directory_structure(base_dir):
    dir_structure = [
        'Yolo_data/training/images', 'Yolo_data/training/labels',
        'Yolo_data/validation/images', 'Yolo_data/validation/labels',
        'Yolo_data/testing/images', 'Yolo_data/testing/labels'
    ]
    for directory in dir_structure:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)



# Function to transform bounding boxes into a normalized format for YOLO
def normalize_bbox(image_size, bbox_coords):
    img_w, img_h = image_size
    x_center = (bbox_coords[0] + bbox_coords[2]) / 2.0
    y_center = (bbox_coords[1] + bbox_coords[3]) / 2.0
    box_w = bbox_coords[2] - bbox_coords[0]
    box_h = bbox_coords[3] - bbox_coords[1]
    return x_center / img_w, y_center / img_h, box_w / img_w, box_h / img_h

# Parse XML annotations and write them to YOLO format
def convert_to_yolo_format(xml_file, label_output, object_classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_size = (int(root.find('size/width').text), int(root.find('size/height').text))
    
    with open(label_output, 'w') as file:
        for obj in root.iter('object'):
            if obj.find('difficult').text == '1':  # Skip difficult objects
                continue
            class_name = obj.find('name').text
            if class_name in object_classes:
                class_idx = object_classes.index(class_name)
                xml_bbox = obj.find('bndbox')
                bbox = (
                    float(xml_bbox.find('xmin').text),
                    float(xml_bbox.find('ymin').text),
                    float(xml_bbox.find('xmax').text),
                    float(xml_bbox.find('ymax').text)
                )
                yolo_bbox = normalize_bbox(image_size, bbox)
                file.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")

# Copy images and process annotations
def prepare_image_data(image_files, img_dir, annot_dir, dest_dir, subset, object_classes):
    for image_file in image_files:
        # Copy images
        shutil.copy(os.path.join(img_dir, image_file), os.path.join(dest_dir, f'Yolo_data/{subset}/images', image_file))

        # Convert and save annotations
        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        label_file = os.path.join(dest_dir, f'Yolo_data/{subset}/labels', os.path.splitext(image_file)[0] + '.txt')
        convert_to_yolo_format(os.path.join(annot_dir, annotation_file), label_file, object_classes)


# Split the Yolo_data and process images for training, validation, and testing
def split_and_prepare_Yolo_data(image_list, img_directory, annot_directory, base_path, object_classes):
    # Split data: 80% train/val, 20% test, then 80/20 split for train and validation sets
    trainval, test_data = train_test_split(image_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(trainval, test_size=0.2, random_state=42)

    # Process images and annotations for each subset
    prepare_image_data(train_data, img_directory, annot_directory, base_path, 'training', object_classes)
    prepare_image_data(val_data, img_directory, annot_directory, base_path, 'validation', object_classes)
    prepare_image_data(test_data, img_directory, annot_directory, base_path, 'testing', object_classes)
    

# Function to generate a YAML configuration file for the Yolo_data
def generate_yaml_config(base_path, object_classes):
    config = {
        'train': os.path.join(base_path, 'Yolo_data/training/images'),
        'val': os.path.join(base_path, 'Yolo_data/validation/images'),
        'test': os.path.join(base_path, 'Yolo_data/testing/images'),
        'nc': len(object_classes),
        'names': object_classes
    }
    with open(os.path.join(base_path, 'Yolo_data', 'data.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)