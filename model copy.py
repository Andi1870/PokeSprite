import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Load the data and label them
def load_data(txt_path, image_folders):
    images = []
    labels = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        filename = line.strip()
        label = filename.split('_')[0]  # z.B. "bulbasaur_Shiny_Generation_2.png" → "bulbasaur"
        
        # Gehe die Sprite-Ordner durch und suche das Bild
        found = False
        for folder in image_folders:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                image = Image.open(path).convert('RGB').resize((96, 96))
                images.append(np.array(image))
                labels.append(label)
                found = True
                break  # Sobald gefunden, nicht weiter suchen

        if not found:
            print(f"Warnung: Datei {filename} nicht gefunden.")
            
    return np.array(images), np.array(labels)

test_folders = ["data/Generation_2", "data/Generation_4"]
train_folders = ["data/Generation_3", "data/Generation_5"]

# Daten laden
test_images, test_labels = load_data("test_data.txt", test_folders)
train_images, train_labels = load_data("train_data.txt", train_folders)

# Label-Encoding, weist jedem Label eine eindeutige Zahl zu
encoder = LabelEncoder()
train_labels_enc = encoder.fit_transform(train_labels)
test_labels_enc = encoder.transform(test_labels)

# One-hot-Encoding, wandelt die Labels in ein binäres Matrixformat um
num_classes = len(encoder.classes_)
train_labels_cat = to_categorical(train_labels_enc, num_classes)
test_labels_cat = to_categorical(test_labels_enc, num_classes)

# Modell erstellen, Dropout erhöht die Robustheit gegen Overfitting
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(0.6),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modell-Übersicht anzeigen
model.summary()

"""
# Datenaugmentation für Trainingsdaten
# Dies hilft, das Modell robuster zu machen, indem es verschiedene Transformationen der Bilder anwendet
datagen = ImageDataGenerator(
    rotation_range=10,           # kleine Drehungen
    width_shift_range=0.05,      # max 5%
    height_shift_range=0.05,
    zoom_range=0.1,              # leichtes Rein-/Rauszoomen
    brightness_range=[0.9, 1.1], # kaum sichtbar heller/dunkler
    horizontal_flip=True,        # manchmal sinnvoll bei Symmetrie
    fill_mode='nearest'
)
"""

"""
checkpoint_cb = ModelCheckpoint(
    filepath="saved_models/pokemon_classifier.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

model = load_model("saved_models/pokemon_classifier.keras")
"""
"""
# Modell trainieren
model.fit(datagen.flow(train_images/255.0, train_labels_cat, batch_size=32),
          validation_data=(test_images/255.0, test_labels_cat), 
          epochs=10)
"""

# === Verbesserte Trainingsstrategie: Data Augmentation & Early Stopping ===
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint(
    filepath="saved_models/pokemon_classifier.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

# Training mit Data Augmentation und Early Stopping
train_gen = datagen.flow(train_images/255.0, train_labels_cat, batch_size=32, subset='training')
val_gen = datagen.flow(train_images/255.0, train_labels_cat, batch_size=32, subset='validation')

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop, checkpoint_cb]
)


# Modell speichern
model.save(os.path.join("saved_models", "pokemon_classifier.keras"))

# Label-Encoder speichern
import pickle
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


# Modell evaluieren
test_loss, test_accuracy = model.evaluate(test_images/255.0, test_labels_cat)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot Beispielbilder
idx = np.random.randint(len(test_images))
img = test_images[idx]
plt.imshow(img)
plt.axis('off')
plt.title(f"Tatsächlich: {test_labels[idx]}")
plt.show()

pred = model.predict(np.expand_dims(img / 255.0, axis=0))
predicted_label = encoder.inverse_transform([np.argmax(pred)])
print(f"Vorhergesagt: {predicted_label[0]} | Tatsächlich: {test_labels[idx]}")