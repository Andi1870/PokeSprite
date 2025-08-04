# Pokémon Sprite Recognition with Machine Learning


This project uses a **Convolutional Neural Network (CNN)** to recognize **Pokémon based on their sprite images**. The dataset includes Pokémon from **Gen 1 to Gen 9**, using both **normal** and **shiny** variants. All sprites are resized to **96x96 pixels** and edited during preprocessing to ensure consistency.

---

## Overview

The goal of this project is to train a machine learning model that can accurately classify Pokémon based on their sprites. Since sprite designs vary between generations and games, the model learns to generalize across different visual styles.

---

## Dataset

- **Sources**: Sprites from Gen 1 - Gen 9 mainline Pokémon game generations (https://www.kaggle.com/datasets/jackemartin/pokemon-sprites)
- **Variants**: Front and back view, battle view and shiny variants
- **Original sizes**: Vary by generation and depending on sprite style
- **Image format**: JPG
- **Preprocessing**:
  - Load the data with `data_loader.py`
  - Check the sizes of the images with `check_sizes.py`
  - See twith`class_distribution.py` (only works after `format_change.py`)
  - To ensure consitency: Resize to **96x96 pixels**, edit the background to white, delete classes with less than 3 images (is needed for splitting the data into training, validation and test sets) and change every image to **RGB** using `format_change.py`
  - Labels will be loaded in the model script

---

## Model

- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow
- **Input**: 96x96 RGB images
- **Output**: Predicted Pokémon name/class

---

## Training
- Train the model using the provided script `model_with_class_weights.py` or `model_with_cropping.py`
- `model_with_class_weights.py` uses the CNN with class weights and without cropping
- `model_with_cropping.py` uses the CNN without class weights and only with cropping
- Options may include batch size, learning rate, optimizer, and epochs

---

## Example Results
- **model_with_class_weights**: `Test Accuracy: 0.7750` `Test Loss: 1.0221`
- **model_with_cropping**: `Test Accuracy: 0.0045` `Test Loss: 12405.3652` (do again)

---

## Structure

- `models/` - Scripts to train a model with class weights or cropping
- `saved_models/` - Can be used to test the model without going through another training (both models are trained with 50 epochs)
- `scripts/` – Python scripts for loading data, changing format, etc.
- `requirements.txt` – List of required Python packages

---

## Setup Instructions

You can set up your environment using `venv`.

---

### Using `venv` (Python 3.10 recommended)

1. **Create virtual environment**:
   ```bash
   python3 -m venv ml-env
   ```

2. **Activate the environment**:
   - On macOS/Linux:
     ```bash
     source ml-env/bin/activate
     ```
   - On Windows:
     ```bash
     .\ml-env\Scripts\activate
     ```

3. **Install requirements**:
   ```bash
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ```
   
---

## Notes

- This repository assumes basic familiarity with Python and Shell comands.
- GPU acceleration is not required but may speed up the training if available (you can use https://colab.google/ for that purpose).

---

## License & Disclaimer

This repository contains **only code** for Pokémon sprite recognition using machine learning.  
It does **not** include any copyrighted Pokémon sprites.

All code is licensed under the MIT License (see [LICENSE](./LICENSE)).  
All Pokémon-related content (names, images, sprites) is the property of **Nintendo**, **Game Freak**, and **The Pokémon Company**, and is used only for **educational and research purposes**.
