import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

# --- Hilfsfunktionen ---

def load_data(txt_path, image_folders):
    images = []
    labels = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        filename = line.strip()
        label = filename.split('_')[0]
        
        found = False
        for folder in image_folders:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                image = Image.open(path).convert('RGB').resize((96, 96))
                images.append(np.array(image))
                labels.append(label)
                found = True
                break

        if not found:
            print(f"Warnung: Datei {filename} nicht gefunden.")
            
    return np.array(images), np.array(labels)

def random_crop(img, crop_size=(72, 72)):
    img = tf.convert_to_tensor(img)
    cropped_img = tf.image.random_crop(img, size=(crop_size[0], crop_size[1], 3))
    return tf.image.resize(cropped_img, [96, 96])

def crop_generator(images, labels, batch_size=32, crop_size=(72, 72)):
    while True:
        idx = np.random.choice(len(images), batch_size)
        batch_x = []
        batch_y = []
        for i in idx:
            cropped = random_crop(images[i] / 255.0, crop_size)
            batch_x.append(cropped.numpy())
            batch_y.append(labels[i])
        yield np.array(batch_x), np.array(batch_y)

def val_generator(images, labels, batch_size=32):
    while True:
        idx = np.random.choice(len(images), batch_size)
        batch_x = images[idx] / 255.0
        batch_y = labels[idx]
        yield batch_x, batch_y

def build_model(input_shape=(96, 96, 3), num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Hauptprogramm ---

# Daten laden
all_folders = ["data/Generation_2", "data/Generation_3", "data/Generation_4", "data/Generation_5"]
all_images, all_labels = load_data("all_data.txt", all_folders)
print(f"Geladene Bilder: {len(all_images)}")

# Labels encodieren
encoder = LabelEncoder()
all_labels_enc = encoder.fit_transform(all_labels)
num_classes = len(encoder.classes_)
all_labels_cat = to_categorical(all_labels_enc, num_classes)

# K-Fold Cross-Validation Setup
k_folds = 5
batch_size = 32
epochs = 10

kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_images, all_labels_enc)):
    print(f"\n🔁 Fold {fold+1}/{k_folds}")

    train_x, val_x = all_images[train_idx], all_images[val_idx]
    train_y, val_y = all_labels_cat[train_idx], all_labels_cat[val_idx]

    model = build_model(input_shape=(96, 96, 3), num_classes=num_classes)

    train_gen = crop_generator(train_x, train_y, batch_size=batch_size, crop_size=(72, 72))
    val_gen = val_generator(val_x, val_y, batch_size=batch_size)

    train_steps = len(train_x) // batch_size
    val_steps = len(val_x) // batch_size

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        verbose=1
    )

    # Evaluieren auf ganzer Val-Menge
    val_eval = model.evaluate(val_x / 255.0, val_y, verbose=0)
    val_accuracy = val_eval[1]
    val_scores.append(val_accuracy)
    print(f"✅ Val-Genauigkeit Fold {fold+1}: {val_accuracy:.4f}")

# Gesamtergebnisse
mean_score = np.mean(val_scores)
std_score = np.std(val_scores)
print(f"\n📊 K-Fold Ergebnis: {mean_score:.4f} ± {std_score:.4f}")

# Optional: Speichere finalen Encoder und Modell (z. B. letztes Modell)
os.makedirs("saved_models", exist_ok=True)
model.save(os.path.join("saved_models", "pokemon_classifier_kfold.keras"))
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Beispiel-Vorhersage
idx = np.random.randint(len(all_images))
img = all_images[idx]
pred = model.predict(np.expand_dims(img / 255.0, axis=0))
predicted_label = encoder.inverse_transform([np.argmax(pred)])

plt.imshow(img)
plt.axis('off')
plt.title(f"Real: {encoder.inverse_transform([np.argmax(all_labels_cat[idx])])[0]} | Predicted: {predicted_label[0]}")
plt.show()
