import os

base_dir = "./data"
generations = ["Generation_2", "Generation_3", "Generation_4", "Generation_5"]

for gen_folder in generations:
    gen_path = os.path.join(base_dir, gen_folder)
    if os.path.isdir(gen_path):
        print(f"Verarbeite Ordner: {gen_folder}")
        for filename in os.listdir(gen_path):
            if filename.endswith(".png") and f"_{gen_folder.lower()}" not in filename.lower():
                # Erstelle den neuen Dateinamen
                name_without_ext, ext = os.path.splitext(filename)
                new_filename = f"{name_without_ext}_{gen_folder}{ext}" 

                old_filepath = os.path.join(gen_path, filename)
                new_filepath = os.path.join(gen_path, new_filename)

                # Umbenennen
                os.rename(old_filepath, new_filepath)
                # print(f"Umbenannt: {filename} -> {new_filename}")
    else:
        print(f"Ordner nicht gefunden: {gen_folder}")

print("Umbenennung abgeschlossen.")