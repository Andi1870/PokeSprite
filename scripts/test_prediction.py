from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Load the trained model and label encoder
model = load_model("saved_models/pokemon_classifier_cropping.keras")

with open("saved_models/label_encoder_cropping.pkl", "rb") as f:
    encoder = pickle.load(f)

# Specify the path to the image you want to predict
image_path = r"data\abra5.jpg"

# Load the image (assuming it's already 96x96 and RGB)
img = Image.open(image_path)
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_index])[0]

print(f"Predicted Pokémon: {predicted_label}")
