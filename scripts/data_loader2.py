import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("jackemartin/pokemon-sprites")
print("Path to dataset files:", path)

# Define the target directory
target_dir = "./data"

# Make sure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Define the generations to copy (only color pixel art generations)
generations_to_copy = ["pokemondb.net"]


# Copy the contents of the downloaded dataset to the target directory
source_gen_path = os.path.join(path)
target_gen_path = os.path.join(target_dir)

# If the target generation directory exists and is not empty, skip copying
if os.path.exists(target_gen_path) and os.listdir(target_gen_path):
    print(f"Skipping copying: Directory already exists and is not empty in '{target_dir}'")

# If the item is a directory, copy it recursively; otherwise, copy the file
if os.path.isdir(source_gen_path):
    shutil.copytree(source_gen_path, target_gen_path, dirs_exist_ok=True)
else:
     shutil.copy2(source_gen_path, target_gen_path)

print(f"✅ Datasets erfolgreich verschoben nach: {target_dir}")