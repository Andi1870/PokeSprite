import os

# Direction to the sprite images
sprite_dir = "./big_data"

all_image_data = []

# Go through all directories and subdirectories in the sprite directory
for root, dirs, files in os.walk(sprite_dir):

    for file_name in files:
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # Store only the file name
            all_image_data.append((file_name))

# Save the file names to text files
with open("big_all_data.txt", "w") as f:
    f.writelines(f"{name}\n" for name in all_image_data)

print(f"\nTraining Images: {len(all_image_data)})")