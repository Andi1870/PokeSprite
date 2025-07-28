from PIL import Image, ImageChops
import os

def is_image_background_white(img: Image.Image) -> bool:
    """
    Prüft, ob der Hintergrund des Bildes komplett weiß ist.
    - Für Bilder ohne Alpha wird geprüft, ob sie komplett weiß sind.
    - Für Bilder mit Alpha wird geprüft, ob transparente Bereiche weiß hinterlegt sind.
    """
    if img.mode != 'RGBA':
        # Kein Alpha-Kanal: Prüfe, ob komplett weiß
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img.convert("RGB"), bg)
        return diff.getbbox() is None
    else:
        alpha = img.split()[-1]
        if alpha.getextrema() == (255, 255):
            # Vollständig undurchsichtig, prüfen ob komplett weiß
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            diff = ImageChops.difference(img, bg)
            return diff.getbbox() is None
        else:
            # Transparente Bereiche vorhanden, prüfen, ob darunter weiß
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            combined = Image.alpha_composite(bg, img)
            diff = ImageChops.difference(combined.convert("RGB"), bg.convert("RGB"))
            return diff.getbbox() is None

def process_single_folder_in_place_to_96x96(input_folder_path):
    target_size = (96, 96)
    supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    print(f"\n--- Beginne In-Place-Verarbeitung für Ordner: {input_folder_path} ---")

    if not os.path.exists(input_folder_path):
        print(f"Fehler: Eingabeordner '{input_folder_path}' existiert nicht. Überspringe.")
        return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filename in os.listdir(input_folder_path):
        if filename.startswith('._') or filename.startswith('~'):
            print(f"   Überspringe temporäre Datei: {filename}")
            continue

        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(input_folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    # Immer in RGBA umwandeln (für Transparenzprüfung & alpha_composite)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')

                    width, height = img.size

                    # Prüfen, ob Bild 96x96 ist UND Hintergrund weiß
                    if (width, height) == target_size and is_image_background_white(img):
                        print(f"   {filename}: Bereits 96x96 und Hintergrund weiß → Überspringe.")
                        skipped_count += 1
                        continue

                    # Bild skalieren & zuschneiden, falls nicht 96x96
                    if (width, height) != target_size:
                        ratio_w = target_size[0] / width
                        ratio_h = target_size[1] / height
                        scale_ratio = max(ratio_w, ratio_h)

                        new_width = int(width * scale_ratio)
                        new_height = int(height * scale_ratio)

                        img = img.resize((new_width, new_height), Image.LANCZOS)

                        left = (new_width - target_size[0]) / 2
                        top = (new_height - target_size[1]) / 2
                        right = (new_width + target_size[0]) / 2
                        bottom = (new_height + target_size[1]) / 2

                        img = img.crop((left, top, right, bottom))

                    # Setze weißen Hintergrund und entferne Transparenz
                    white_bg = Image.new("RGB", target_size, (255, 255, 255))
                    white_bg.paste(img, (0, 0), img)

                    white_bg.save(filepath)
                    print(f"   Verarbeitet & weißer Hintergrund gesetzt: {filename}")
                    processed_count += 1

            except Exception as e:
                print(f"   Fehler beim Verarbeiten von {filename}: {e}")
                error_count += 1
        else:
            print(f"   Überspringe Datei (nicht unterstütztes Format): {filename}")
            skipped_count += 1

    print(f"--- Verarbeitung für Ordner '{input_folder_path}' abgeschlossen. ---")
    print(f"   Verarbeitet: {processed_count} | Übersprungen: {skipped_count} | Fehler: {error_count}")

# Beispielaufruf (bitte anpassen auf deine Ordnerstruktur):
base_input_path = "./data"
folders_to_process = [
    "Generation_2",
    "Generation_3",
    "Generation_4",
    "Generation_5"
]

print("--- START: Vorbereitung der Bilder (Originale werden überschrieben!) ---")
print("!!! SICHERHEITSHINWEIS: Bitte stelle sicher, dass du ein Backup deiner Originalbilder hast. !!!")

for folder_name in folders_to_process:
    current_input_folder = os.path.join(base_input_path, folder_name)
    process_single_folder_in_place_to_96x96(current_input_folder)

print("--- ENDE: Bildvorbereitung abgeschlossen! ---")
