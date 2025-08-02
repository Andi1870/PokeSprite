import os
import sys
import re
from collections import Counter

def find_low_count_classes(image_folders, min_count_to_print=2):
    """
    Searches specified folders, counts the number of images per class,
    and outputs classes that fall below a certain threshold.

    Args:
        image_folders (list): A list of paths to the folders containing the images.
        min_count_to_print (int): The threshold. Classes with fewer images
                                  will be output.

    Returns:
        dict: A dictionary containing the class names and their frequency.
    """
    class_counts = Counter()
    total_files = 0

    # The regular expression that captures everything before the first digit
    label_pattern = re.compile(r'^([a-zA-Z]+)')

    # Iterate through each folder in the list
    for folder in image_folders:
        if not os.path.isdir(folder):
            print(f"Warning: Folder '{folder}' not found. Skipping.", file=sys.stderr)
            continue

        print(f"Searching folder: {folder}")

        for filename in os.listdir(folder):
            # Ignore hidden files and folders
            if filename.startswith('.') or os.path.isdir(os.path.join(folder, filename)):
                continue

            # Extract the class name from the filename
            match = label_pattern.match(filename)
            if match:
                class_name = match.group(1).lower()  # Convert class name to lowercase
                class_counts[class_name] += 1
                total_files += 1
            else:
                print(f"Warning: Filename '{filename}' does not match the expected format. Skipping.")

    if not class_counts:
        print("No classes found. Check the folder paths or the naming format of the files.")
        return {}

    print(f"\n--- Class Frequencies ---")
    print(f"Total number of processed files: {total_files}")
    print(f"Number of found classes: {len(class_counts)}")
    print("-" * 25)

    # Sort classes by frequency
    sorted_counts = sorted(class_counts.items(), key=lambda item: item[1])

    low_count_classes = []

    # Apply logic to output classes with fewer than 'min_count_to_print'
    for class_name, count in sorted_counts:
        if count < min_count_to_print:
            low_count_classes.append((class_name, count))

    if low_count_classes:
        print(f"\n--- Classes with fewer than {min_count_to_print} images: ---")
        for class_name, count in low_count_classes:
            print(f"  - Class '{class_name}': {count} image(s)")
    else:
        print(f"\nNo classes found with fewer than {min_count_to_print} images. All good.")

    return class_counts

# Define the folders to check
image_folders_to_check = "./big_data"

# Set the minimum value for the number of images per class
MINIMUM_IMAGES_PER_CLASS = 3

# Call the function to find and print low count classes
find_low_count_classes(image_folders_to_check, min_count_to_print=MINIMUM_IMAGES_PER_CLASS)