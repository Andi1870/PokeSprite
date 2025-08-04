import os
import random
import numpy as np
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# For reproducibility
random.seed(42)  
np.random.seed(42)
tf.random.set_seed(42)

# Load the data and label them
def load_data(data_folder):
    images = []
    labels = []
    
    # regex to extract the class name: everything before the first digit
    label_pattern = re.compile(r'^([a-zA-Z]+)')

    if not os.path.isdir(data_folder):
        print(f"Error: Directory '{data_folder}' not found.")
        return np.array([]), np.array([])
    
    for filename in os.listdir(data_folder):
        # Ignore hidden files or non-image files
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            continue

        match = label_pattern.match(filename)
        if match:
            label = match.group(1).lower()
            
            path = os.path.join(data_folder, filename)
            try:
                # Load image
                image = Image.open(path)
                images.append(np.array(image))
                labels.append(label)
            except Exception as e:
                print(f"Warning: File {filename} could not be loaded. Error: {e}")
        else:
            print(f"Warning: Filename '{filename}' does not match the expected pattern. Skipping.")

    return np.array(images), np.array(labels)

# Random Cropping: Crop random section and rescale to 96x96
def random_crop(img, crop_size=(72, 72)):
    img = tf.convert_to_tensor(img)
    cropped_img = tf.image.random_crop(img, size=(crop_size[0], crop_size[1], 3))
    return tf.image.resize(cropped_img, [96, 96])

# Generator that produces crops (for training)
def crop_generator(images, labels, batch_size=32, crop_size=(72, 72)):
    # The generator randomly selects images from the entire dataset
    while True:
        idx = np.random.choice(len(images), batch_size)
        batch_x = []
        batch_y = []
        for i in idx:
            # Here the cropping and normalization takes place
            cropped = random_crop(images[i] / 255.0, crop_size)
            batch_x.append(cropped.numpy())
            batch_y.append(labels[i])
        yield np.array(batch_x), np.array(batch_y)

# Validation Generator (without Cropping)
def val_generator(images, labels, batch_size=32):
    while True:
        for i in range(0, len(images), batch_size):
            batch_x = images[i:i+batch_size] / 255.0
            batch_y = labels[i:i+batch_size]
            yield batch_x, batch_y


data_folder = "./data"

# Load all data
all_images, all_labels = load_data(data_folder)
print(f"Loaded images: {len(all_images)}")
print(f"Loaded labels: {len(all_labels)}")

# Label-Encoding all labels
encoder = LabelEncoder()
all_labels_enc = encoder.fit_transform(all_labels)

num_classes = len(encoder.classes_)

# Split in Train (70%) and Temp (30%)
train_images, temp_images, train_labels_enc, temp_labels_enc = train_test_split(
    all_images, all_labels_enc, test_size=0.3, random_state=42, stratify=all_labels_enc)

# Temp split into Validation (15%) and Test (15%)
val_images, test_images, val_labels_enc, test_labels_enc = train_test_split(
    temp_images, temp_labels_enc, test_size=0.5, random_state=42, stratify=None)

# One-hot-Encoding for all three splits
train_labels_cat = to_categorical(train_labels_enc, num_classes)
val_labels_cat = to_categorical(val_labels_enc, num_classes)
test_labels_cat = to_categorical(test_labels_enc, num_classes)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

# Create model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Define optimizer
optimizer = Adam(learning_rate=0.001)

# Compile model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Generators creation
train_gen = crop_generator(train_images, train_labels_cat, batch_size=32, crop_size=(72, 72))
val_gen = val_generator(val_images, val_labels_cat, batch_size=32)

# Steps per epoch calculation
train_steps_per_epoch = len(train_images) // 32
val_steps_per_epoch = len(val_images) // 32

print(f"Training steps per epoch: {train_steps_per_epoch}")
print(f"Validation steps per epoch: {val_steps_per_epoch}")

# Train model
model.fit(
    train_gen,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps_per_epoch,
    epochs=50
)

# Save model
os.makedirs("saved_models", exist_ok=True)
model.save(os.path.join("saved_models", "pokemon_classifier_cropping.keras"))

# Save label encoder
with open("saved_models/label_encoder_cropping.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Pick a random image from the test set and predict
idx = np.random.randint(len(test_images))
img = test_images[idx]
pred = model.predict(np.expand_dims(img / 255.0, axis=0))
predicted_label = encoder.inverse_transform([np.argmax(pred)])

plt.imshow(img)
plt.axis('off')
plt.title(f"Real: {encoder.inverse_transform([test_labels_enc[idx]])[0]} | Predicted: {predicted_label[0]}")
plt.show()