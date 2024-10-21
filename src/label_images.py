import cv2
import os
import re
import shutil
import random
import yaml


class ImageLabeler:
    def __init__(self, config):
        self.class_list = config.get('classes', [])
        self.class_color = [tuple(color) for color in config.get('colors', [])]
        self.drawing, self.ix, self.iy = False, -1, -1
        self.cx, self.cy = -1, -1  # Track current mouse x and y coordinates
        self.drawing_bbox = None
        self.bbox_list, self.bbox_class_name_list = [], []
        self.default_color = (0, 0, 0)  # Black color for unconfirmed bounding boxes
        self.panel_width, self.display_size = 350, (1000, 1000)
        self.background_color = (208, 253, 255)  # Cream background color
        self.data_ind = 0
        self.is_confirmed = False

        if len(self.class_list) > 20:
            print(f"Error: Maximum of 20 classes supported. Trimming to 20.")
            self.class_list = self.class_list[:20]
            self.class_color = self.class_color[:20]

        if len(self.class_list) != len(self.class_color):
            raise ValueError("The number of classes must match the number of colors.")

        self.dataset_handler = DatasetManager(config['split'])
        self.dataset_handler.create_folders(config['unlabeled_folder'])

    def extract_number(self, filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else 0

    def draw_bbox(self, event, x, y, flags, param):
        """Handles drawing of bounding boxes and tracking of mouse movements."""
        self.cx, self.cy = x, y  # Update current mouse coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = self._clip_to_image_bounds(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x, y = self._clip_to_image_bounds(x, y)
            self.drawing_bbox = (self.ix, self.iy, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x, y = self._clip_to_image_bounds(x, y)
            self.drawing_bbox = (self.ix, self.iy, x, y)

    def label_images(self, image_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        self.image_paths = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder)
                                   if img.lower().endswith((".jpg", ".png"))], key=self.extract_number)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.draw_bbox)

        self.data_ind = 0
        while self.data_ind < len(self.image_paths):
            img_path = self.image_paths[self.data_ind]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue

            img_display, scale_x, scale_y, offset_x, offset_y = self._resize_with_aspect_ratio(img)
            self.original_img_dims = (img.shape[1], img.shape[0])  # (img_width, img_height)
            self.display_dims = (img_display.shape[1], img_display.shape[0])  # (display_width, display_height)

            # Check if the image is already confirmed
            
            self.is_confirmed = self._check_image_confirmed(img_path)
             

            while True:
                self._render_image(img_display, scale_x, scale_y, offset_x, offset_y)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):  # Quit the program on 'q'
                    print("Quitting program.")
                    cv2.destroyAllWindows()
                    return

                elif key == 13:  # Enter key
                    if self.bbox_list:  # Only save if there are bounding boxes
                        self._save_and_move_to_next(img_path, img, scale_x, scale_y, offset_x, offset_y)
                        break  # Move to the next image

                elif key == ord("n"):  # Next key
                    self._move_next()
                    break
                elif key == ord("b"):  # Back key
                    self._move_back()
                    break

                # Handle class selection and other key presses
                if key != -1 and key != 255:
                    self._handle_keypress(key, img_path, scale_x, scale_y, offset_x, offset_y)

        cv2.destroyAllWindows()

    def _resize_with_aspect_ratio(self, img):
        """Resize the image while maintaining aspect ratio and add padding."""
        img_h, img_w = img.shape[:2]
        scale = min(self.display_size[0] / img_w, self.display_size[1] / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        #padding = [(self.display_size[1] - new_h) // 3, (self.display_size[0] - new_w) // 3]
        padding = [0, 0]
        padded_img = cv2.copyMakeBorder(resized_img, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT, value=self.background_color)
        return padded_img, scale, scale, padding[1], padding[0]

    def _clip_to_image_bounds(self, x, y):
        """Ensure that the bounding box coordinates stay within the image boundaries."""
        x = max(0, min(self.display_dims[0], x))  # Clip to width
        y = max(0, min(self.display_dims[1], y))  # Clip to height
        return x, y

    def _render_image(self, img_display, scale_x, scale_y, offset_x, offset_y):
        display_copy = img_display.copy()

        self._draw_guidelines(display_copy)
        if self.drawing_bbox:
            self._draw_bbox(display_copy, self.drawing_bbox, self.default_color)

        # Draw the bounding boxes for confirmed images
        if self.is_confirmed:
            label_file_path = self._get_label_file_path(self.image_paths[self.data_ind])
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as label_file:
                    for line in label_file:
                        parts = line.strip().split()
                        class_index = int(parts[0])
                        center_x, center_y, width, height = map(float, parts[1:])

                        ix = int((center_x - width / 2) * img_display.shape[1])
                        iy = int((center_y - height / 2) * img_display.shape[0])
                        x = int((center_x + width / 2) * img_display.shape[1])
                        y = int((center_y + height / 2) * img_display.shape[0])
                        
                        bbox = (ix, iy, x, y)
                        class_name = self.class_list[class_index]
                        self._draw_bbox(display_copy, bbox, self.class_color[self.class_list.index(class_name)])

        for bbox, class_name in zip(self.bbox_list, self.bbox_class_name_list):
            scaled_bbox = [int(coord * scale + offset) for coord, scale, offset in zip(bbox, (scale_x, scale_y, scale_x, scale_y), (offset_x, offset_y, offset_x, offset_y))]
            self._draw_bbox(display_copy, scaled_bbox, self.class_color[self.class_list.index(class_name)])

        cv2.imshow("image", self._add_panel(display_copy))

    def _get_label_file_path(self, img_path):
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        for split, split_path in self.dataset_handler.folders.items():
            label_path = os.path.join(split_path, 'labels', label_filename)
            if os.path.exists(label_path):
                return label_path
        return None

    def _draw_guidelines(self, img):
        """Draw guidelines at the current mouse position."""
        cv2.line(img, (0, self.cy), (self.display_size[0], self.cy), (128, 128, 128), 1)
        cv2.line(img, (self.cx, 0), (self.cx, self.display_size[1]), (128, 128, 128), 1)

    def _draw_bbox(self, img, bbox, color):
        """Draw a bounding box."""
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def _add_panel(self, img_display):
        """Adds a side panel listing class names, key bindings, and confirmation status."""
        panel = cv2.copyMakeBorder(img_display, 0, 0, 0, self.panel_width, cv2.BORDER_CONSTANT, value=self.background_color)

        for idx, class_name in enumerate(self.class_list):
            text = f"{idx + 1 if idx < 10 else 'Shift + ' + str(idx - 9)}: {class_name}"
            cv2.putText(panel, text, (img_display.shape[1] + 10, 30 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        keys = ["Enter", "n", "b", "esc", "backspace"]
        binds = ["confirm", "next", "back", "remove bbs", "remove last bb"]
        for idx2 in range(len(keys)):
            text = f"{keys[idx2]}: {binds[idx2]}"
            cv2.putText(panel, text, (img_display.shape[1] + 10, 30 + (idx+idx2 + 2) * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        status_text = "Confirmed" if self.is_confirmed else "Unconfirmed"
        status_text = f"Status: {status_text}"
        
        cv2.putText(panel, status_text, (img_display.shape[1] + 10, 30 + (len(self.class_list) + len(keys) + 1) * 40 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        return panel

    def _handle_keypress(self, key, img_path, scale_x, scale_y, offset_x, offset_y):
        """Handle keypress events for selecting classes and removing bounding boxes with backspace."""
        
        if key == 8:  # Backspace key
            if self.bbox_list:
                removed_bbox = self.bbox_list.pop()
                removed_class = self.bbox_class_name_list.pop()
                print(f"Removed {removed_class} bounding box: {removed_bbox}")
            else:
                print("No bounding boxes to remove.")
        else:
            self._delete_confirmed_image(img_path)
            class_index = self._get_class_index(key)
            if class_index is None:
                return

            if class_index >= len(self.class_list):
                print("Index " + str(class_index), " out of range.")

            class_name = self.class_list[class_index]
            if self.drawing_bbox:
                # Reverse the scaling and offset applied during rendering, clip to the image size
                ix = max(0, min(self.original_img_dims[0], (self.drawing_bbox[0] - offset_x) / scale_x))
                iy = max(0, min(self.original_img_dims[1], (self.drawing_bbox[1] - offset_y) / scale_y))
                x = max(0, min(self.original_img_dims[0], (self.drawing_bbox[2] - offset_x) / scale_x))
                y = max(0, min(self.original_img_dims[1], (self.drawing_bbox[3] - offset_y) / scale_y))

                self.bbox_list.append((ix, iy, x, y))
                self.bbox_class_name_list.append(class_name)
                print(f"Added {class_name}")
                self.drawing_bbox = None  # Reset after adding bbox

    def _delete_confirmed_image(self, img_path):
        self.is_confirmed = False
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        path_exists = False
        for split, split_path in self.dataset_handler.folders.items():
            label_path = os.path.join(split_path, 'labels', label_filename)
            path_exists = os.path.exists(label_path)
            if path_exists:
                img_path = os.path.join(split_path, 'images', img_filename)
                os.remove(img_path)
                os.remove(label_path)

    def _save_and_move_to_next(self, img_path, img, scale_x, scale_y, offset_x, offset_y):
        """Save annotations and move to the next image."""
        if self.is_confirmed:
            print(f"Overwrite existing labels for {os.path.basename(img_path)}")
        else:
            print(f"Saved labels for {os.path.basename(img_path)}")
        
        self.dataset_handler.save_image_and_labels(img_path, img, self._generate_label_data(img.shape[1], img.shape[0]))
        self.bbox_list.clear()
        self.bbox_class_name_list.clear()
        self.data_ind += 1

    def _move_next(self):
        self.bbox_list.clear()
        self.bbox_class_name_list.clear()
        self.data_ind += 1

    def _move_back(self):
        if self.data_ind > 0:
            self.bbox_list.clear()
            self.bbox_class_name_list.clear()
            self.data_ind -= 1

    def _get_class_index(self, key):
        """Get class index based on key input."""
        if 49 <= key <= 57:
            return key - 49  # Number keys 1-9
        if key == 48:
            return 9  # Number key 0
        if key == 33:  # Shift + 1 ('!')
            return 10
        if key == 64:  # Shift + 2 ('@')
            return 11
        if key == 94:  # Shift + 6 ('^')
            return 15
        if key == 38:  # Shift + 7 ('&')
            return 16
        if key == 42:  # Shift + 8 ('*')
            return 17
        if key == 40:  # Shift + 9 ('(')
            return 18
        if key == 41:  # Shift + 0 (')')
            return 19
        if 35 <= key <= 41:
            return key - 35 + 12  # Shift + 3 to Shift + 0
        print(f"Invalid class key: {key}. Available keys: 1 to 9, 0, Shift + 1 to Shift + 0")
        return None

    def _generate_label_data(self, img_width, img_height):
        return [f"{self.class_list.index(class_name)} {(ix + x) / 2 / img_width} {(iy + y) / 2 / img_height} {abs(x - ix) / img_width} {abs(y - iy) / img_height}\n"
                for (ix, iy, x, y), class_name in zip(self.bbox_list, self.bbox_class_name_list)]

    def _check_image_confirmed(self, img_path):
        """Get the path to the label file for the image."""
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        path_exists = False
        for split, split_path in self.dataset_handler.folders.items():
            label_path = os.path.join(split_path, 'labels', label_filename)
            cur_path_exists = os.path.exists(label_path)
            if cur_path_exists:
                path_exists = True
        return path_exists

class DatasetManager:
    def __init__(self, split_ratio):
        self.split_ratio = split_ratio
        self.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    def create_folders(self, labeled_folder):
        self.labeled_folder = os.path.join(self.data_folder, 'labeled', labeled_folder)
        self.folders = {split: os.path.join(self.labeled_folder, split) for split in ['train', 'val', 'test']}
        for folder in self.folders.values():
            os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
            os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

    def save_image_and_labels(self, img_path, img, label_data):
        folder_type = random.choices(['train', 'val', 'test'], self.split_ratio)[0]
        img_dest = os.path.join(self.folders[folder_type], 'images', os.path.basename(img_path))
        label_dest = os.path.join(self.folders[folder_type], 'labels', f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
        shutil.copy(img_path, img_dest)
        with open(label_dest, 'w') as label_file:
            label_file.writelines(label_data)
        print(f"Image and labels saved in {folder_type} folder.")

    def generate_yolo_yaml(self, labeled_folder, class_list):
        yaml_path = os.path.join(self.labeled_folder, f"data.yaml")
        yaml_content = {'train': 'train/images', 'val': 'val/images', 'test': 'test/images', 'nc': len(class_list), 'names': class_list}
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(yaml_content, yaml_file)
        print(f"Generated YOLO YAML file at {yaml_path}")


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config('config.yaml')
    unlabeled_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'unlabeled', config['unlabeled_folder'])
    output_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'labeled', config['unlabeled_folder'])

    labeler = ImageLabeler(config)
    labeler.label_images(unlabeled_folder, output_folder)
    labeler.dataset_handler.generate_yolo_yaml(config['unlabeled_folder'], config['classes'])


if __name__ == "__main__":
    main()
