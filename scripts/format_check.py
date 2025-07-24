from PIL import Image
import os

folder_path = 'data\Generation_3'  # Ersetze dies durch den tatsächlichen Pfad zu deinem Ordner
image_formats = {}

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png'):
        filepath = os.path.join(folder_path, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                image_formats[filename] = f"{width}x{height}"
        except Exception as e:
            print(f"Fehler beim Öffnen von {filename}: {e}")

# Ergebnisse ausgeben
print("Pixelformate der PNG-Dateien:")
for filename, dimensions in image_formats.items():
    print(f"{filename}: {dimensions}")

# Überprüfen, ob alle das gleiche Format haben
if len(set(image_formats.values())) == 1 and image_formats:
    print("\nAlle PNG-Dateien haben dasselbe Pixelformat.")
elif not image_formats:
    print("\nKeine PNG-Dateien im angegebenen Ordner gefunden.")
else:
    print("\nDie PNG-Dateien haben unterschiedliche Pixelformate.")
    unique_formats = set(image_formats.values())
    print("Gefundene Formate:", unique_formats)