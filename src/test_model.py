import os
import cv2
import numpy as np
import ultralytics
import argparse
from matplotlib import pyplot as plt
from ultralytics.utils import ops
import yaml

def load_image(image_path):
    """Load an image from file."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img  # Keep BGR format

def load_labels(label_path):
    """Load YOLO format labels from file."""
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            labels.append((class_id, x_center, y_center, width, height))
    return labels

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def main():
    config = load_config('config.yaml')
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on test set and display results.')
    parser.add_argument('--model_id', type=str, help='model ID', default="steve")
    parser.add_argument('--dataset_id', type=str, help='Dataset identifier.', default="fruit_big")
    args = parser.parse_args()

    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_folder = os.path.join(base_dir, 'data', 'labeled', args.dataset_id)
    test_images_path = os.path.join(dataset_folder, 'test', 'images')
    test_labels_path = os.path.join(dataset_folder, 'test', 'labels')
    model_path = os.path.join(base_dir, 'src', 'runs', 'detect', f'{args.model_id}', 'weights', 'best.pt')

    # Load model
    model = ultralytics.YOLO(model_path)

    # Get image and label file names
    image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg') or f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(test_labels_path) if f.endswith('.txt')])

    for img_file, lbl_file in zip(image_files, label_files):
        img_path = os.path.join(test_images_path, img_file)
        lbl_path = os.path.join(test_labels_path, lbl_file)

        # Load and process image
        image = load_image(img_path)
        labels = load_labels(lbl_path)

        # Predict with the model
        predictions = model.predict(img_path, imgsz=320, verbose=False)
        
        # Initialize bounding boxes
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                confidence = box.conf.cpu().item()  # Get confidence score
                # bounding format in [x, y, width, height]
                box_cord = box.xywh[0].cpu().numpy()  # Get bounding box coordinates
                box_label = box.cls  # class label of the box

                bounding_boxes.append([prediction.names[int(box_label)], confidence, np.asarray(box_cord)])
                
        # Draw bounding boxes on the image
        for bbox in bounding_boxes:
            # Translate bounding box info back to the format of [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(bbox[2])  # bbox[2] contains the [x, y, width, height]
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]  # Ensure coordinates are integers

            # Draw bounding box
            col = config['colors'][config['classes'].index(bbox[0])]
            col_t = (int(col[2]), int(col[1]), int(col[0]))  # Convert to BGR format
            
            if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
                # Ensure the image is a valid format
                image = cv2.rectangle(image, (x1, y1), (x2, y2), col_t, thickness=2)
                # Draw class label with confidence
                label_text = f"{bbox[0]}: {bbox[1]:.2f}"  # bbox[0] is the label, bbox[1] is the confidence
                image = cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_t, 2)
            else:
                print(f"Invalid image format: {image.shape if image is not None else 'None'}")

        # Display the image with bounding boxes
        cv2.imshow('YOLO Detection', image)
        key = cv2.waitKey(0)  # Wait for a key press to close the image window
        cv2.destroyAllWindows()
        if key == ord("q"):
            break
        

if __name__ == '__main__':
    main()