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
  - To ensure consistency: Resize to **96x96 pixels**, edit the background to white, delete classes with less than 3 images (is needed for splitting the data into training, validation and test sets) and change every image to **RGB** using `format_change.py`
  - See the overall class distribution with `class_distribution.py` (only works after `format_change.py`)
  - Labels will be loaded in the model script

---

## Model

- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow
- **Input**: 96x96 RGB images
- **Output**: Predicted Pokémon name/class

---

## Training
- Train the model using the provided script `model_with_class_weights.py`
- `model_with_class_weights.py` uses the CNN with class weights
- Options may include batch size, learning rate, optimizer, and epochs

---

## Example Results
- **model_with_class_weights**: `Test Accuracy: 0.8722` `Test Loss: 0.7378`

---

## Structure

- `models/` - Scripts to train a model with class weights
- `saved_models/` - Can be used to test the model without going through another training (trained with 100 epochs)
- `scripts/` – Python scripts for loading data, changing format, etc.
- `requirements.txt` – List of required Python packages

---

## Setup Instructions

You can set up your environment using `venv`.


### Using `venv` (Python 3.10 recommended)

1. **Create virtual environment**:
   ```bash
   python3.10 -m venv venv
   ```

2. **Activate the environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install requirements**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**Make sure that the venv is selected as the interpreter**

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
