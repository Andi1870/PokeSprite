import os
import re
from PIL import Image, ImageChops, ImageOps
from collections import Counter

def is_image_background_white(img: Image.Image) -> bool:
    """
    Check if the image has a white background.
    This function works for both RGB and RGBA images.
    It returns True if the image is completely white or has a white background,
    and False otherwise.
    
    Args:
        img (Image.Image): The image to check.

    Returns:
        bool: True if the image has a white background, False otherwise.
    """
    if img.mode != 'RGBA':
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img.convert("RGB"), bg)
        return diff.getbbox() is None
    else:
        alpha = img.split()[-1]
        if alpha.getextrema() == (255, 255):
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            diff = ImageChops.difference(img, bg)
            return diff.getbbox() is None
        else:
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            combined = Image.alpha_composite(bg, img)
            diff = ImageChops.difference(combined.convert("RGB"), bg.convert("RGB"))
            return diff.getbbox() is None

def process_and_clean_single_folder(input_folder_path, min_members=3):
    target_size = (96, 96)
    supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    pad_color = (255, 255, 255)

    print(f"\n--- Start the data cleaning and processing for folder: {input_folder_path} ---")

    if not os.path.exists(input_folder_path):
        print(f"Error: Input folder '{input_folder_path}' does not exist. Skipping.")
        return

    # Phase 1: Analyze class frequencies
    class_counts = Counter()
    label_pattern = re.compile(r'^([a-zA-Z]+)')
    
    all_files_to_check = [f for f in os.listdir(input_folder_path) if f.lower().endswith(supported_formats)]

    for filename in all_files_to_check:
        match = label_pattern.match(filename)
        if match:
            class_name = match.group(1).lower()
            class_counts[class_name] += 1

    # Identify classes to be deleted
    low_count_classes = {class_name for class_name, count in class_counts.items() if count < min_members}
    
    if not low_count_classes:
        print(f"  No classes found with fewer than {min_members} members. No images will be deleted.")
    else:
        print(f"  The following classes with fewer than {min_members} images will be removed:")
        for class_name in sorted(list(low_count_classes)):
            print(f"    - Class '{class_name}': {class_counts[class_name]} image(s)")


    # Phase 2: Processing & Deletion
    processed_count = 0
    deleted_count = 0
    skipped_count = 0
    error_count = 0
    
    for filename in os.listdir(input_folder_path):
        filepath = os.path.join(input_folder_path, filename)

        # Skip temporary or unsupported files
        if filename.startswith('._') or filename.startswith('~') or not filename.lower().endswith(supported_formats):
            skipped_count += 1
            continue

        match = label_pattern.match(filename)
        if not match:
            print(f"  Warning: Filename '{filename}' does not match the pattern. Skipping.")
            skipped_count += 1
            continue
        
        class_name = match.group(1).lower()

        # Deletion logic
        if class_name in low_count_classes:
            try:
                os.remove(filepath)
                print(f"  > Deleted: {filename} (Class '{class_name}')")
                deleted_count += 1
            except OSError as e:
                print(f"  ! Error deleting {filepath}: {e}")
            continue

        # Processing logic (only for classes that are kept)
        try:
            with Image.open(filepath) as img:
                original_size = img.size

                # Always convert to RGBA for alpha channel checking
                img = img.convert('RGBA')

                # Check if processing is necessary
                if original_size == target_size and img.getchannel('A').getextrema() == (255, 255):
                    print(f"  {filename}: Correct size & opaque → Skipping.")
                    skipped_count += 1
                    continue

                # Create a new image with a white background in the target size
                new_img = Image.new('RGB', target_size, pad_color)

                # Insert the scaled image into the center
                if original_size != target_size:
                    img = ImageOps.contain(img, target_size)
                        
                paste_position = (
                    (target_size[0] - img.width) // 2,
                    (target_size[1] - img.height) // 2
                )
                
                new_img.paste(img, paste_position, img)
                
                new_img.save(filepath)
                print(f"  Processed: {filename}")
                processed_count += 1

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            error_count += 1

    print(f"\n--- Processing for folder '{input_folder_path}' completed. ---")
    print(f"  Processed: {processed_count} | Deleted: {deleted_count} | Skipped: {skipped_count} | Errors: {error_count}")


base_input_path = "./data"

print("--- START: Data Cleaning & Image Preparation ---")

# Execute the function for the base path
process_and_clean_single_folder(base_input_path, min_members=3)

print("--- END: Image Preparation Completed! ---")