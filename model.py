import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split # Diese Zeile entfernen wir
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

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

# Random Cropping: Crop zufälligen Ausschnitt und rescale auf 96x96
def random_crop(img, crop_size=(72, 72)):
    img = tf.convert_to_tensor(img)
    cropped_img = tf.image.random_crop(img, size=(crop_size[0], crop_size[1], 3))
    return tf.image.resize(cropped_img, [96, 96])

# Generator, der Crops erzeugt (für Training)
def crop_generator(images, labels, batch_size=32, crop_size=(72, 72)):
    # Der Generator wählt zufällig Bilder aus den gesamten Daten aus
    while True:
        idx = np.random.choice(len(images), batch_size)
        batch_x = []
        batch_y = []
        for i in idx:
            # Hier findet das Cropping und die Normalisierung statt
            cropped = random_crop(images[i] / 255.0, crop_size)
            batch_x.append(cropped.numpy())
            batch_y.append(labels[i])
        yield np.array(batch_x), np.array(batch_y)

# Validation Generator (ohne Cropping)
def val_generator(images, labels, batch_size=32):
    # Der Generator wählt zufällig Bilder aus den gesamten Daten aus
    while True:
        idx = np.random.choice(len(images), batch_size)
        # Hier findet nur die Normalisierung statt, kein Cropping
        batch_x = images[idx] / 255.0
        batch_y = labels[idx]
        yield batch_x, batch_y


all_folders = ["data/Generation_2", "data/Generation_3", "data/Generation_4", "data/Generation_5"]

# Alle Daten laden
all_images, all_labels = load_data("all_data.txt", all_folders)
print(f"Geladene Bilder: {len(all_images)}")
print(f"Geladene Labels: {len(all_labels)}")

# Label-Encoding auf alle Labels
encoder = LabelEncoder()
all_labels_enc = encoder.fit_transform(all_labels)

# One-hot-Encoding
num_classes = len(encoder.classes_)
all_labels_cat = to_categorical(all_labels_enc, num_classes)


# Modell erstellen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Optimizer definieren
optimizer = Adam(learning_rate=0.001)

# Modell kompilieren
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Generatoren erstellen - jetzt beide basierend auf den kompletten Daten
train_gen = crop_generator(all_images, all_labels_cat, batch_size=32, crop_size=(72, 72))
val_gen = val_generator(all_images, all_labels_cat, batch_size=32)


# Gesamtanzahl der Bilder
total_samples = len(all_images)
batch_size = 32

# Angenommen, du möchtest 80% der Daten für "Trainingsschritte" und 20% für "Validierungsschritte" nutzen.
train_steps_per_epoch = int(0.8 * total_samples) // batch_size
val_steps_per_epoch = int(0.2 * total_samples) // batch_size

print(f"Trainingsschritte pro Epoche: {train_steps_per_epoch}")
print(f"Validierungsschritte pro Epoche: {val_steps_per_epoch}")


# Modell trainieren
model.fit(
    train_gen,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps_per_epoch,
    epochs=100
)

# Modell speichern
model.save(os.path.join("saved_models", "pokemon_classifier.keras"))

# Label-Encoder speichern
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


# Beispielbild mit Vorhersage plotten
# Wähle ein zufälliges Bild aus den gesamten Daten für die Vorhersage
idx = np.random.randint(len(all_images))
img = all_images[idx]
pred = model.predict(np.expand_dims(img / 255.0, axis=0))
predicted_label = encoder.inverse_transform([np.argmax(pred)])

plt.imshow(img)
plt.axis('off')
plt.title(f"Real: {encoder.inverse_transform([np.argmax(all_labels_cat[idx])])[0]} | Predicted: {predicted_label[0]}")
plt.show()