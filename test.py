# Count how many image files are in the folder
from PIL import Image
import os
folder_path = 'data/Generation_5'  # Replace with the actual path to your folder
image_count = sum(1 for filename in os.listdir(folder_path) if filename.lower().endswith('.png'))
print(f"Anzahl der PNG-Dateien im Ordner '{folder_path}': {image_count}")

