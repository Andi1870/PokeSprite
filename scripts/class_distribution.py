import collections
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from PIL import Image


# Load the data and label them
def load_data(data_folder):
    images = []
    labels = []
    
    # regex to extract the class name: everything before the first digit
    label_pattern = re.compile(r'^([a-zA-Z]+)')

    if not os.path.isdir(data_folder):
        print(f"Error: Folder '{data_folder}' not found.")
        return np.array([]), np.array([])
    
    for filename in os.listdir(data_folder):
        # Ignore hidden files or non-image files
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            continue

        match = label_pattern.match(filename)
        if match:
            label = match.group(1).lower()
            
            path = os.path.join(data_folder, filename)
            try:
                # Load image
                image = Image.open(path)
                images.append(np.array(image))
                labels.append(label)
            except Exception as e:
                print(f"Warning: File {filename} could not be loaded. Error: {e}")
        else:
            print(f"Warning: Filename '{filename}' does not match the expected pattern. Skipping.")

    return np.array(images), np.array(labels)

data_folder = "./data"

# Load all data
all_images, all_labels = load_data(data_folder)
print(f"Loaded images: {len(all_images)}")
print(f"Loaded labels: {len(all_labels)}")

# Count occurrences of each class
counter = collections.Counter(all_labels)
plt.bar(counter.keys(), counter.values())
plt.xticks(rotation=90)
plt.title("Number of Images per Class")
plt.show()