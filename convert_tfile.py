import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("model/plant_cancer_yesno.h5")

# Convert to TF Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save file
with open("model/plant_cancer_yesno.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved model/plant_cancer_yesno.tflite")
