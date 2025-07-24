from PIL import Image
import os

def process_single_folder_in_place_to_96x96(input_folder_path):
    """
    Verarbeitet alle Bilder in einem spezifischen Ordner direkt, skaliert sie auf 96x96 Pixel
    und überschreibt die Originaldateien.
    """
    target_size = (96, 96)
    # Sicherheitsschritt: Nur PNG Formate vorhanden
    supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    print(f"\n--- Beginne In-Place-Verarbeitung für Ordner: {input_folder_path} ---")

    # Überprüfen, ob der Ordner überhaupt existiert
    if not os.path.exists(input_folder_path):
        print(f"Fehler: Eingabeordner '{input_folder_path}' existiert nicht. Überspringe.")
        return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filename in os.listdir(input_folder_path):
        # Ignoriere temporäre Dateien, die von einigen Programmen erstellt werden können
        if filename.startswith('._') or filename.startswith('~'):
            print(f"   Überspringe temporäre Datei: {filename}")
            continue

        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(input_folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    # Konvertiere das Bild bei Bedarf in RGB, um Probleme mit bestimmten Formaten zu vermeiden
                    # und um einheitliche Speicherung zu gewährleisten (z.B. für JPG)
                    if img.mode in ('RGBA', 'P', 'CMYK'):
                        img = img.convert('RGB')

                    width, height = img.size

                    # Prüfe, ob das Bild bereits die Zielgröße hat
                    if width == target_size[0] and height == target_size[1]:
                        print(f"   {filename}: Bereits 96x96. Überspringe Verarbeitung.")
                        skipped_count += 1
                        continue

                    # Bestimme das Skalierungsverhältnis, um sicherzustellen, dass das Bild die Zielgröße bedeckt
                    ratio_w = target_size[0] / width
                    ratio_h = target_size[1] / height

                    # Wähle das größere Verhältnis, um sicherzustellen, dass das Bild die Zielabmessungen überlappt
                    scale_ratio = max(ratio_w, ratio_h)

                    new_width = int(width * scale_ratio)
                    new_height = int(height * scale_ratio)

                    # Skaliere das Bild, LANCZOS wird fürs Upscaling empfohlen (Balance zwischen Qualität und Rechenzeit)
                    img = img.resize((new_width, new_height), Image.LANCZOS)

                    # Berechne den Zuschneidebereich (Mittelzuschnitt)
                    left = (new_width - target_size[0]) / 2
                    top = (new_height - target_size[1]) / 2
                    right = (new_width + target_size[0]) / 2
                    bottom = (new_height + target_size[1]) / 2

                    # Schneide das Bild zu
                    img = img.crop((left, top, right, bottom))

                    # Speichere das Bild über die Originaldatei
                    img.save(filepath)

                    print(f"   Verarbeitet und überschrieben: {filename} ({width}x{height} -> 96x96)")
                    processed_count += 1

            except Exception as e:
                print(f"   Fehler beim Verarbeiten von {filename}: {e}")
                error_count += 1
        else:
            print(f"   Überspringe Datei (nicht unterstütztes Format): {filename}")
            skipped_count += 1 # Zähle auch nicht unterstützte Formate als übersprungen

    print(f"--- Verarbeitung für Ordner '{input_folder_path}' abgeschlossen. ---")
    print(f"   Verarbeitet: {processed_count} | Übersprungen: {skipped_count} | Fehler: {error_count}")

# Basispfad des Eingabeordners
base_input_path = "./data"

# Liste der Ordnernamen, die verarbeitet werden sollen
folders_to_process = [
    "Generation_2",
    "Generation_3",
    "Generation_4",
    "Generation_5"
]

print("--- START: Vorbereitung der Bilder (Originale werden überschrieben!) ---")
print("!!! SICHERHEITSHINWEIS: Bitte stelle sicher, dass du ein Backup deiner Originalbilder hast. !!!")

# Iteriere durch jeden Ordner und verarbeite ihn
for folder_name in folders_to_process:
    current_input_folder = os.path.join(base_input_path, folder_name)
    process_single_folder_in_place_to_96x96(current_input_folder)
print("--- ENDE: Bildvorbereitung abgeschlossen! ---")