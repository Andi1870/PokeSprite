import os
import random
import numpy as np
import pickle
import re
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model and label encoder
model = load_model("saved_models/pokemon_classifier_class_weights.keras")

with open("saved_models/label_encoder_class_weights.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load all image files
data_folder = "./data"
image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Select 10 random images
random_files = random.sample(image_files, 10)

# Prepare figure
plt.figure(figsize=(15, 6))

for i, file in enumerate(random_files):
    image_path = os.path.join(data_folder, file)

    # Load and preprocess image
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0
    input_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(input_array, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = encoder.inverse_transform([predicted_index])[0]

    # Extract true label from filename
    match = re.match(r'^([a-zA-Z]+)', file)
    true_label = match.group(1).lower() if match else "(unbekannt)"

    # Set title color: green = correct, red = wrong
    title_color = 'green' if predicted_label == true_label else 'red'

    # Plot image
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"T: {true_label}\nP: {predicted_label}", color=title_color)

plt.tight_layout()
plt.show()