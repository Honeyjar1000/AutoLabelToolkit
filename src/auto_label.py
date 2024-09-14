import os
import cv2
import numpy as np
import ultralytics
import yaml
import random
from ultralytics.utils import ops

def save_labels(label_path, boxes):
    """Save YOLO format labels to a file."""
    with open(label_path, 'w') as f:
        for box in boxes:
            class_id, xywh, confidence = box
            f.write(f"{class_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def create_directories(base_path, splits):
    """Create directories for train, val, and test splits."""
    for split in splits:
        for sub_dir in ['images', 'labels']:
            path = os.path.join(base_path, split, sub_dir)
            os.makedirs(path, exist_ok=True)

def save_yaml(config, dataset_path):
    """Save YAML configuration for the labeled dataset."""
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    config = load_config('config.yaml')

    # Define paths
    model_path = os.path.join(script_dir, 'runs', 'detect', f'{config["model"]}', 'weights', 'best.pt')
    unlabeled_folder = os.path.join(base_dir, 'data', 'unlabeled', config["unlabeled_folder"])
    labeled_folder = os.path.join(base_dir, 'data', 'labeled', config["unlabeled_folder"])

    splits = ['train', 'val', 'test']
    split_ratios = config['split']
    
    create_directories(labeled_folder, splits)

    # Load the model
    model = ultralytics.YOLO(model_path)

    # Get image file names
    image_files = [f for f in os.listdir(unlabeled_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Shuffle images and split based on ratios
    random.shuffle(image_files)
    total_images = len(image_files)
    train_end = int(total_images * split_ratios[0])
    val_end = train_end + int(total_images * split_ratios[1])
    
    split_images = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    for split, files in split_images.items():
        for img_file in files:
            img_path = os.path.join(unlabeled_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            # Get original image dimensions
            orig_height, orig_width = img.shape[:2]
            
            # Perform predictions
            results = model.predict(img_path, imgsz=320, verbose=False)
            
            bounding_boxes = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf.cpu().item()  # Get confidence score
                    xyxy = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                    class_id = int(box.cls)  # Class label of the box
                    
                    # Calculate YOLO format bounding boxes
                    x_min, y_min, x_max, y_max = xyxy
                    x_center = ((x_min + x_max) / 2) / orig_width
                    y_center = ((y_min + y_max) / 2) / orig_height
                    width = (x_max - x_min) / orig_width
                    height = (y_max - y_min) / orig_height
                    
                    bounding_boxes.append([class_id, [x_center, y_center, width, height], confidence])
            
            # Save images and labels
            output_image_path = os.path.join(labeled_folder, split, 'images', img_file)
            output_label_path = os.path.join(labeled_folder, split, 'labels', os.path.splitext(img_file)[0] + '.txt')

            # Save the image
            cv2.imwrite(output_image_path, img)
            print(f"Saved image to: {output_image_path}")

            # Save labels
            save_labels(output_label_path, bounding_boxes)
            print(f"Saved labels to: {output_label_path}")

    # Create YAML file
    dataset_config = {
        'train': {
            'images': 'train/images',
            'labels': 'train/labels'
        },
        'val': {
            'images': 'val/images',
            'labels': 'val/labels'
        },
        'test': {
            'images': 'test/images',
            'labels': 'test/labels'
        },
        'nc': len(config['classes']),
        'names': config['classes']
    }
    save_yaml(dataset_config, labeled_folder)

if __name__ == '__main__':
    main()
