import random
import os

# Direction to the sprite images
sprite_dir = "./data"

generation_folders = ['Generation_2', 'Generation_3', 'Generation_4', 'Generation_5']

all_image_data = []

# Go through all directories and subdirectories in the sprite directory
for root, dirs, files in os.walk(sprite_dir):

    # Extract the generation folder name from the path
    folder_name = os.path.basename(root)

    # Check if the current folder belongs to our relevant generations
    if folder_name in generation_folders:
        print(f"Processing folder: {folder_name}")
        for file_name in files:
            if file_name.endswith(".png"):
                # Store only the file name
                all_image_data.append((file_name))

# Save the file names to text files
with open("all_data.txt", "w") as f:
    f.writelines(f"{name}\n" for name in all_image_data)

print(f"\nTraining Images: {len(all_image_data)} (from folders: {generation_folders})")