from PIL import Image
import os
from pillow_heif import register_heif_opener

# Register the HEIF opener
register_heif_opener()


def resize_image(input_path, output_path, target_size):
    """
    Resize the image to the target size and save it to the output path.

    :param input_path: Path to the input image
    :param output_path: Path to save the resized image
    :param target_size: Tuple of (width, height) to resize the image to
    """
    with Image.open(input_path) as img:
        # Resize the image
        img_resized = img.resize(target_size, Image.LANCZOS)

        # Save the resized image
        img_resized.save(output_path, format='PNG')
        print(f"Resized and saved: {output_path}")


def resize_images_in_folder(folder_path, target_size):
    """
    Resize all images in the given folder to the target size and replace the originals.

    :param folder_path: Path to the folder containing the images
    :param target_size: Tuple of (width, height) to resize the images to
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            input_path = os.path.join(folder_path, filename)
            output_path = input_path  # Replace the original image
            resize_image(input_path, output_path, target_size)

# Path to the folder containing the images
folder_path = 'my_test/input'

# Target size
target_size = (880, 670)

# Resize all images in the folder
resize_images_in_folder(folder_path, target_size)