from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Modell und LabelEncoder laden
model = load_model("saved_models/pokemon_classifier.keras")

with open("saved_models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Bildpfad
image_path = r"data\Generation_3\bellsprout_Normal_Generation_3.png"

# Bild laden (wenn es schon 96x96 und RGB ist)
img = Image.open(image_path)
img_array = np.array(img) / 255.0  # nur noch normalisieren
img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen

# Vorhersage machen
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_index])[0]

print(f"Vorhergesagtes Pokémon: {predicted_label}")
