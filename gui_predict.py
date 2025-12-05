"""
GUI predictor for plant cancer yes/no model.

- Looks for the trained model under ./model/
  (tries plant_cancer_yesno.keras first, then plant_cancer_yesno.h5)
- Opens a Tkinter window with a "Choose Image" button.
- After you pick an image, it shows:
    CANCER (xx.xx%) or HEALTHY (xx.xx%)

Run from the repo root like this:

    python gui_predict.py
"""

import tkinter as tk
from tkinter import filedialog, Label, messagebox
from pathlib import Path
import os
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2


# -----------------------------------------------------------
# CONFIG / PATHS
# -----------------------------------------------------------

# Folder that this script lives in (your repo root)
SCRIPT_DIR = Path(__file__).resolve().parent

# Model directory and possible filenames
MODEL_DIR = SCRIPT_DIR / "model"
POSSIBLE_MODEL_NAMES = [
    "plant_cancer_yesno.keras",  # new Keras format
    "plant_cancer_yesno.h5",     # older H5 format
]

# Try to find an existing model file
MODEL_PATH = None
for name in POSSIBLE_MODEL_NAMES:
    candidate = MODEL_DIR / name
    if candidate.exists():
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    # If we get here, training either didn't run or saved to a different name
    raise SystemExit(
        "ERROR: Could not find a trained model file in ./model/\n"
        "Expected one of:\n"
        + "\n".join(f"  - {name}" for name in POSSIBLE_MODEL_NAMES)
        + "\n\nRun train.py first, then try again."
    )

# Class names must match your training folders (alphabetical by folder name)
CLASS_NAMES = ["cancer", "healthy"]

# Image size used when training the model (see train.py)
IMG_SIZE = (224, 224)


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------

try:
    # safe_mode=False is important for some models saved with newer TF/Keras
    model = keras.models.load_model(MODEL_PATH, safe_mode=False)
except Exception as e:
    raise SystemExit(f"ERROR loading model from {MODEL_PATH}:\n{e}")


# -----------------------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------------------

def predict_image(path: str) -> str:
    """
    Load an image from `path`, preprocess it exactly like in training,
    run the model, and return a nicely formatted result string.
    """
    # Load and resize
    img = image.load_img(path, target_size=IMG_SIZE)

    # Convert to array of shape (224, 224, 3)
    x = image.img_to_array(img)

    # Add batch dimension -> shape (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Use the same preprocessing as MobileNetV2 training
    x = mobilenet_v2.preprocess_input(x)

    # Run model prediction
    preds = model.predict(x)[0]  # vector of probabilities, length 2

    # Index of the highest probability
    idx = int(np.argmax(preds))

    predicted_class = CLASS_NAMES[idx].upper()
    confidence = float(preds[idx]) * 100.0

    return f"{predicted_class} ({confidence:.2f}%)"


# -----------------------------------------------------------
# TKINTER GUI CALLBACKS
# -----------------------------------------------------------

def choose_file() -> None:
    """
    Open a file dialog, let the user pick an image, then display
    the prediction in the result label.
    """
    path = filedialog.askopenfilename(
        title="Select plant image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    # If the user canceled
    if not path:
        return

    if not os.path.isfile(path):
        messagebox.showerror("File Error", "The selected file does not exist.")
        return

    try:
        result = predict_image(path)
        label_result.config(text=result)
    except Exception as e:
        # Show any prediction error in a popup
        messagebox.showerror("Prediction Error", str(e))


# -----------------------------------------------------------
# BUILD THE TKINTER WINDOW
# -----------------------------------------------------------

root = tk.Tk()
root.title("Plant Cancer Yes/No Detector")

# Title label
label_title = Label(
    root,
    text="Plant Cancer Yes/No Detector",
    font=("Arial", 16)
)
label_title.pack(pady=10)

# Button to choose an image
button_choose = tk.Button(
    root,
    text="Choose Image",
    font=("Arial", 12),
    command=choose_file
)
button_choose.pack(pady=10)

# Label to show prediction result
label_result = Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 14)
)
label_result.pack(pady=10)

# Start the GUI event loop
root.mainloop()
