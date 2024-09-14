import cv2
import os
import re  # For regular expressions
import yaml


class LabelVisualizer:
    def __init__(self, labeled_folder, class_list, class_color):
        self.labeled_folder = labeled_folder
        self.splits = ['train', 'val', 'test']  # Define dataset splits
        self.class_list = class_list
        self.class_color = class_color  # Example colors for classes

    def extract_number(self, filename):
        """Sort images by number."""
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else 0

    def load_labels(self, label_file, img_width, img_height):
        """Load bounding box labels from a YOLO format text file."""
        bbox_list = []
        with open(label_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                class_idx = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                bbox_list.append({
                    'class': self.class_list[class_idx],
                    'bbox': [x_center, y_center, width, height]
                })
        return bbox_list

    def draw_bbox(self, img, bbox, color):
        """Draw a single bounding box on the image."""
        img_height, img_width = img.shape[:2]
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def count_images_and_labels(self):
        """Count total images and labels in each split."""
        total_images = 0
        label_count = {class_name: 0 for class_name in self.class_list}
        split_data = {split: {'images': 0, 'labels': 0} for split in self.splits}

        for split in self.splits:
            image_folder = os.path.join(self.labeled_folder, f'{split}/images')
            label_folder = os.path.join(self.labeled_folder, f'{split}/labels')

            if not os.path.exists(image_folder):
                continue

            image_paths = [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.png'))]
            total_images += len(image_paths)
            split_data[split]['images'] = len(image_paths)

            for img_name in image_paths:
                label_file = os.path.join(label_folder, f"{os.path.splitext(img_name)[0]}.txt")
                if os.path.exists(label_file):
                    with open(label_file, 'r') as file:
                        lines = file.readlines()
                        split_data[split]['labels'] += len(lines)
                        for line in lines:
                            class_idx = int(line.strip().split()[0])
                            class_name = self.class_list[class_idx]
                            label_count[class_name] += 1

        return total_images, label_count, split_data

    def show_images(self):
        """Display images with bounding boxes."""
        image_folder = os.path.join(self.labeled_folder, 'train/images')  # Defaulting to 'train/images'
        label_folder = os.path.join(self.labeled_folder, 'train/labels')  # Defaulting to 'train/labels'

        image_paths = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) 
                              if img.lower().endswith(('.jpg', '.png'))], key=self.extract_number)

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue

            img_height, img_width = img.shape[:2]
            label_file = os.path.join(label_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
            
            if not os.path.exists(label_file):
                print(f"No labels found for {img_path}")
                continue

            bbox_list = self.load_labels(label_file, img_width, img_height)
            for bbox_data in bbox_list:
                class_name = bbox_data['class']
                bbox = bbox_data['bbox']
                color = self.class_color[self.class_list.index(class_name)]
                self.draw_bbox(img, bbox, color)

            cv2.imshow('Labeled Image', img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit on 'q'
                break
            elif key == 32:  # Spacebar to go to the next image
                continue

        cv2.destroyAllWindows()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config('config.yaml')

    # Folder paths
    labeled_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'labeled', config['unlabeled_folder'])

    # Initialize LabelVisualizer with the labeled folder and class list
    visualizer = LabelVisualizer(labeled_folder, config['classes'], config['colors'])

    # Print dataset information
    total_images, label_count, split_data = visualizer.count_images_and_labels()
    
    # Display data summary
    print(f"Total images: {total_images}")
    print("Label counts:")
    for label, count in label_count.items():
        print(f"  {label}: {count}")
    
    print("\nData split:")
    for split, data in split_data.items():
        print(f"  {split.capitalize()} - Images: {data['images']}, Labels: {data['labels']}")

    # Start showing images with bounding boxes
    visualizer.show_images()


if __name__ == "__main__":
    main()