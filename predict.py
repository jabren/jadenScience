import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# -----------------------------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------------------------
model = keras.models.load_model("model/plant_cancer_yesno.h5")

# Must match your folder names
class_names = ["cancer", "healthy"]

img_size = (224, 224)

def predict_image(path):
    # Load image and resize
    img = image.load_img(path, target_size=img_size)
    
    # Convert to array
    x = image.img_to_array(img)
    
    # Expand dimensions to match model input
    x = np.expand_dims(x, axis=0)
    
    # Preprocess same way as training
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Run prediction
    preds = model.predict(x)[0]
    
    # Get highest confidence index
    idx = np.argmax(preds)
    
    print("----------------------------")
    print(f"Prediction: {class_names[idx].upper()}")
    print(f"Confidence: {preds[idx]*100:.2f}%")
    print("----------------------------")

# EXAMPLE USAGE:
predict_image("test.jpg")
