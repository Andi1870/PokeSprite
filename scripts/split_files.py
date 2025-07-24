import random
import os

# Direction to the sprite images
sprite_dir = "./data"

train_generations_folders = ['Generation_3', 'Generation_5']
test_generations_folders = ['Generation_2', 'Generation_4']

train_data = []
test_data = []

all_image_data = []

# Go through all directories and subdirectories in the sprite directory
for root, dirs, files in os.walk(sprite_dir):

    # Extract the generation folder name from the path
    folder_name = os.path.basename(root)

    # Check if the current folder belongs to our relevant generations
    if folder_name in train_generations_folders or folder_name in test_generations_folders:
        print(f"Processing folder: {folder_name}")
        for file_name in files:
            if file_name.endswith(".png"):
                # Store only the file name and the generation folder
                all_image_data.append((file_name, folder_name))

# Shuffle the collected paths to ensure a random distribution within the generations
random.seed(42) # for reproducibility
random.shuffle(all_image_data)

# Split the images based on their assigned generation folders
for file_name, folder_name in all_image_data:
    if folder_name in train_generations_folders:
        train_data.append(file_name) # Add only the file name
    elif folder_name in test_generations_folders:
        test_data.append(file_name) # Add only the file name

# Save the file names to text files
with open("train_data.txt", "w") as f:
    f.writelines(f"{name}\n" for name in train_data)

with open("test_data.txt", "w") as f:
    f.writelines(f"{name}\n" for name in test_data)

print(f"\nTraining Images: {len(train_data)} (from folders: {train_generations_folders})")
print(f"Test Images: {len(test_data)} (from folders: {test_generations_folders})")