import tkinter as tk
from tkinter import filedialog, Label, messagebox
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2
import numpy as np
import os

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

# Path to your saved model
MODEL_PATH = "model/plant_cancer_yesno.h5"

# Class names must match how you trained the model
# (in alphabetical order by folder name)
class_names = ["cancer", "healthy"]

# Image size used during training
img_size = (224, 224)

# -------------------------------------------------------
# LOAD MODEL (with basic error handling)
# -------------------------------------------------------
try:
    model = keras.models.load_model(MODEL_PATH, safe_mode=False)

except OSError:
    # If the model file is missing, show a popup and exit
    raise SystemExit(
        f"Could not find model file at '{MODEL_PATH}'. "
        "Make sure you trained the model and the path is correct."
    )

# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------

def predict_image(path: str) -> str:
    """
    Load an image from 'path', preprocess it the same way
    as during training, run the model, and return a nice
    formatted string with the predicted class and confidence.
    """
    # Load image and resize to the expected size
    img = image.load_img(path, target_size=img_size)

    # Convert the image to a NumPy array
    x = image.img_to_array(img)

    # Add batch dimension (model expects shape: (1, 224, 224, 3))
    x = np.expand_dims(x, axis=0)

    # Preprocess using the same function as MobileNetV2 training
    x = mobilenet_v2.preprocess_input(x)

    # Run prediction (returns an array of probabilities)
    preds = model.predict(x)[0]  # shape: (2,)

    # Get index of the highest probability
    idx = np.argmax(preds)

    # Format the result nicely
    predicted_class = class_names[idx].upper()
    confidence = preds[idx] * 100.0
    return f"{predicted_class} ({confidence:.2f}%)"

# -------------------------------------------------------
# FILE CHOOSER CALLBACK
# -------------------------------------------------------

def choose_file():
    """
    Open a file dialog, let user pick an image, then
    show prediction in the label.
    """
    # Ask for a file
    path = filedialog.askopenfilename(
        title="Select plant image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    # If the user canceled, path will be empty
    if not path:
        return

    # Optional: Check the file actually exists
    if not os.path.isfile(path):
        messagebox.showerror("Error", "Selected file does not exist.")
        return

    try:
        # Run prediction
        result = predict_image(path)
        # Update the GUI label with result
        label_result.config(text=result)
    except Exception as e:
        # If something goes wrong, show an error popup
        messagebox.showerror("Prediction Error", str(e))

# -------------------------------------------------------
# TKINTER GUI SETUP
# -------------------------------------------------------

root = tk.Tk()
root.title("Plant Cancer Detector")

# Main title label
label_title = Label(
    root,
    text="Plant Cancer Yes/No Detector",
    font=("Arial", 16)
)
label_title.pack(pady=10)

# Button to pick image
button = tk.Button(
    root,
    text="Choose Image",
    font=("Arial", 12),
    command=choose_file
)
button.pack(pady=10)

# Label where prediction result will appear
label_result = Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 14)
)
label_result.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
