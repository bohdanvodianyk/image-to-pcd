import os
from PIL import Image
from pillow_heif import register_heif_opener

# Register the HEIF opener
register_heif_opener()

# Path to the directory containing HEIC images
directory_path = 'my_test/input/'

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.lower().endswith('.heic'):
        # Construct full file path
        heic_file_path = os.path.join(directory_path, filename)

        # Open the HEIC image
        image = Image.open(heic_file_path)

        # Create the output PNG file path
        png_file_path = os.path.join(directory_path, os.path.splitext(filename)[0] + '.png')

        # Save the image as PNG
        image.save(png_file_path, format='PNG')

        # Remove the original HEIC file
        os.remove(heic_file_path)

        print(f"Converted and removed: {filename}")

print("All HEIC files have been converted to PNG and the originals have been removed.")
