import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("jackemartin/pokemon-sprites")
print("Path to dataset files:", path)

# Define the target directory
target_dir = "./big_data"

# Make sure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Go into Sprites directory
path = os.path.join(path, "pokemon_images/pokemondb.net")

# If the target generation directory exists and is not empty, skip copying
if os.path.exists(target_dir) and os.listdir(target_dir):
    print(f"Skipping copying: Directory already exists and is not empty in '{target_dir}'")

# If the item is a directory, copy it recursively; otherwise, copy the file
if os.path.isdir(path):
    shutil.copytree(path, target_dir, dirs_exist_ok=True)
else:
    shutil.copy2(path, target_dir)

print(f"Dataset copied to: {target_dir}")