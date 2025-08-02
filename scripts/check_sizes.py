import os
from PIL import Image

def count_image_sizes(folder_path):
    """
    Counts the frequency of different image sizes in a folder.

    Args:
        folder_path (str): The path to the folder containing the images.

    Returns:
        dict: A dictionary where the keys are the image sizes as tuples (width, height)
        and the values are the count of images with that size.
    """
    size_counts = {}

    # Check if the directory exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return None

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Build the full path to the file
        file_path = os.path.join(folder_path, filename)

        # Skip directories and only process files
        if os.path.isdir(file_path):
            continue

        try:
            # Open the image with Pillow (Pillow is tolerant of file formats)
            with Image.open(file_path) as img:
                # Get the size (width, height) of the image
                size = img.size

                # Count the frequency of the size in the dictionary
                if size in size_counts:
                    size_counts[size] += 1
                else:
                    size_counts[size] = 1
        except Exception as e:
            # Give a warning if the file is not an image or is corrupted
            print(f"Warning: '{filename}' could not be processed as an image. Error: {e}")

    return size_counts


# Define the image folder
image_folder = "./big_data"

# Call the counting function
sizes = count_image_sizes(image_folder)
    
if sizes:
    print(f"Found image sizes in '{image_folder}':")

    # Sort the results for better readability
    sorted_sizes = sorted(sizes.items(), key=lambda item: item[1], reverse=True)
        
    for size, count in sorted_sizes:
        print(f"  - Size {size[0]}x{size[1]}: {count} images")

    # Print the total number of processed images
    total_images = sum(sizes.values())
    print(f"\nTotal number of processed images: {total_images}")