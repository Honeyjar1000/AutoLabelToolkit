import os
import cv2
import re

def crop_to_square(image):
    """
    Crop an image to a square by centering the crop.
    
    :param image: The input image to crop.
    :return: Cropped square image.
    """
    height, width = image.shape[:2]
    min_dim = min(height, width)

    # Calculate the center coordinates
    center_x, center_y = width // 2, height // 2

    # Define the cropping box
    x1 = max(center_x - min_dim // 2, 0)
    y1 = max(center_y - min_dim // 2, 0)
    x2 = min(center_x + min_dim // 2, width)
    y2 = min(center_y + min_dim // 2, height)

    # Crop and return the square image
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def get_next_image_number(subfolder_path):
    """
    Find the next available image number in the subfolder to avoid name conflicts.
    
    :param subfolder_path: Path to the subfolder containing images.
    :return: The next available image number.
    """
    existing_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    highest_number = 0
    for file in existing_files:
        match = re.match(rf'{re.escape(subfolder_path)}_(\d+)\.jpg$', file)
        if match:
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number
    return highest_number + 1

def resize_and_rename_images_in_folder(base_folder, target_size=(100, 100)):
    """
    Resize all images in the specified folder and its subfolders to the target size, if they are not already the correct size.
    Rename the images sequentially in each subfolder.

    :param base_folder: Path to the base folder containing subfolders with images.
    :param target_size: Desired size to resize images (width, height).
    """
    target_width, target_height = target_size

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_counter = get_next_image_number(subfolder_path)  # Initialize counter based on existing files
            for image_name in os.listdir(subfolder_path):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subfolder_path, image_name)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        print(f"Failed to load image {image_path}")
                        continue
                    
                    img_height, img_width = image.shape[:2]
                    
                    # Crop the image to a square
                    squared_image = crop_to_square(image)
                    
                    # Resize the squared image
                    resized_image = cv2.resize(squared_image, target_size)
                    
                    # Generate new image name
                    new_image_name = f"{subfolder}_{image_counter}.jpg"  # Sequential naming without leading zeros
                    new_image_path = os.path.join(subfolder_path, new_image_name)
                    
                    # Save the resized image with the new name
                    cv2.imwrite(new_image_path, resized_image)
                    print(f"Resized and saved {new_image_path}")
                    
                    # Remove the old image
                    os.remove(image_path)
                    
                    image_counter += 1  # Increment counter for next image

if __name__ == "__main__":
    # Get the absolute path to the parent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset = 'example_animals'
    base_folder = os.path.join(base_dir, 'data', 'unlabeled', dataset)  # Path to the base folder with subfolders
    
    resize_and_rename_images_in_folder(base_folder, target_size=(100, 100))