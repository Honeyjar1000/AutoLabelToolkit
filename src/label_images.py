import cv2
import os
import re
import shutil
import random

# Initialize global variables
drawing = False
ix, iy = -1, -1
drawing_bbox = None
color = (255, 0, 0)  # Default color for drawing
confirmed_color = (0, 255, 0)
image_paths = []  # List to store image paths
current_index = 0  # To track the current image index
cx, cy = -1, -1
data_split = [0.7, 0.2, 0.1]

bbox_list = []
bbox_class_name_list = []
class_list = ["cat", "dog", "fish"]
class_color = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]

def extract_number(filename):
    """Extract numerical part from filename for sorting."""
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else 0

def draw_bbox(event, x, y, flags, param):
    global ix, iy, cx, cy, drawing, bbox_list, bbox_class_name_list, class_color, color, drawing_bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        # When left mouse button is pressed, clear unconfirmed bounding boxes
        bbox_list[:] = [bbox for bbox in bbox_list if bbox != drawing_bbox]
        bbox_class_name_list[:] = [name for i, name in enumerate(bbox_class_name_list) if bbox_list[i] != drawing_bbox]
        drawing = True
        ix, iy = x, y
        drawing_bbox = (ix, iy, ix, iy)  # Initialize the current drawing bbox
    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y
        if drawing:
            # Update the current drawing bbox coordinates
            drawing_bbox = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Finalize the current drawing bbox coordinates
        drawing_bbox = (ix, iy, x, y)

def save_image_and_labels(img_path, output_folder, label_file_path):
    """Saves the image and corresponding labels to the appropriate folder."""
    # Determine which subset (train/val/test) this image should go to
    split_choice = random.choices(['train', 'val', 'test'], data_split, k=1)[0]

    # Copy the image to the appropriate subfolder in the 'images' directory
    img_subfolder = os.path.join(output_folder, 'images', split_choice)
    if not os.path.exists(img_subfolder):
        os.makedirs(img_subfolder)
    shutil.copy(img_path, img_subfolder)

    # Move the label file to the corresponding 'labels' subfolder
    label_subfolder = os.path.join(output_folder, 'labels', split_choice)
    if not os.path.exists(label_subfolder):
        os.makedirs(label_subfolder)
    
    # Define the destination path for the label file
    destination_label_file = os.path.join(label_subfolder, os.path.basename(label_file_path))
    
    # If the destination file already exists, remove it
    if os.path.exists(destination_label_file):
        os.remove(destination_label_file)

    # Move the label file to the destination
    shutil.move(label_file_path, destination_label_file)

def label_images(image_folder, output_folder):
    global img, color, bbox_list, bbox_class_name_list, class_color, drawing_bbox, current_index, img_width, img_height, new_width, new_height

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    class_index = 0
    color = (255, 0, 0)
    bbox_list = []
    bbox_class_name_list = []

    global image_paths
    image_paths = sorted(
        [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder) if img_name.lower().endswith((".jpg", ".png"))],
        key=lambda x: extract_number(os.path.basename(x))
    )

    new_width, new_height = 1000, 1000  # Desired display dimensions

    while 0 <= current_index < len(image_paths):
        img_path = image_paths[current_index]
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image {img_path}")
            current_index += 1
            continue

        img_height, img_width = img.shape[:2]
        drawing_bbox = None

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_bbox)

        while True:
            img_display = cv2.resize(img, (new_width, new_height))

            # Draw grey lines from window edges to mouse position
            cv2.line(img_display, (0, cy), (new_width, cy), (128, 128, 128), 1)
            cv2.line(img_display, (cx, 0), (cx, new_height), (128, 128, 128), 1)

            # Draw the current bounding box being drawn
            if drawing_bbox:
                top_left = (int(drawing_bbox[0]), int(drawing_bbox[1]))
                bottom_right = (int(drawing_bbox[2]), int(drawing_bbox[3]))
                cv2.rectangle(img_display, top_left, bottom_right, color, 2)

            # Draw confirmed bounding boxes
            for (ix, iy, x, y), class_name in zip(bbox_list, bbox_class_name_list):
                top_left = (int(ix), int(iy))
                bottom_right = (int(x), int(y))
                cv2.rectangle(img_display, top_left, bottom_right, class_color[class_list.index(class_name)], 2)

            cv2.imshow("image", img_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to quit
                print("ESC pressed. Exiting.")
                cv2.destroyAllWindows()
                return

            elif key == 49:  # 1
                class_name = class_list[0]
                if drawing_bbox:
                    bbox_list.append(drawing_bbox)
                    bbox_class_name_list.append(class_name)
                    drawing_bbox = None
                    print("Added ", class_name)
            elif key == 50:  # 2
                class_name = class_list[1]
                if drawing_bbox:
                    bbox_list.append(drawing_bbox)
                    bbox_class_name_list.append(class_name)
                    drawing_bbox = None
                    print("Added ", class_name)
            elif key == 51:  # 3
                class_name = class_list[2]
                if drawing_bbox:
                    bbox_list.append(drawing_bbox)
                    bbox_class_name_list.append(class_name)
                    drawing_bbox = None
                    print("Added ", class_name)

            elif key == 8:  # Backspace key to remove the last bounding box
                if bbox_list:
                    bbox_list.pop()
                    name = bbox_class_name_list.pop()
                    print("Removed " + name + " bounding box")

            elif key == ord('b'):  # B key to go back to the previous image
                if current_index > 0:
                    current_index -= 1
                break
            elif key == 13:  # Enter key to go to the next image
                # Save labels in YOLO format
                if bbox_list:
                    label_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
                    with open(label_file_path, "w") as label_file:
                        for (ix, iy, x, y), class_name in zip(bbox_list, bbox_class_name_list):
                            # Convert bounding box to YOLO format
                            x_center = ((ix + x) / 2.0) / img_width
                            y_center = ((iy + y) / 2.0) / img_height
                            w = abs(x - ix) / img_width
                            h = abs(y - iy) / img_height
                            class_index = class_list.index(class_name)
                            label_file.write(f"{class_index} {x_center} {y_center} {w} {h}\n")
                    print(f"Saved annotations to {label_file_path}")

                    save_image_and_labels(img_path, output_folder, label_file_path)

                cv2.destroyAllWindows()
                current_index += 1
                bbox_list = []
                bbox_class_name_list = []
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Dataset split ratios: train, val, test
    
    unlabeled_folder = 'D:/yolo_object_detection/data/unlabeled'
    labeled_folder = 'D:/yolo_object_detection/data/labeled'

    # Define the specific folder within unlabeled_folder you want to label
    image_folder = os.path.join(unlabeled_folder, 'example_dog')
    
    # Define the output folder where labeled data will be saved
    output_folder = os.path.join(labeled_folder, 'example_dog')

    label_images(image_folder, output_folder)