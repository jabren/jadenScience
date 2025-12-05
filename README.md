# 🌿 Plant Cancer Detection – Machine Learning Science Project

This project uses **TensorFlow** and **MobileNetV2** to classify plant leaf images as either:

- **CANCER**
- **HEALTHY**

You can train the model on your own dataset and use a simple **Tkinter GUI** to make predictions on new images.

---

# 📂 Project Structure

```
project-root/
│
├── train.py                # Train the model
├── gui_predict.py          # Tkinter GUI for predictions
├── predict.py              # CLI-based prediction script (optional)
│
├── model/                  # Saved models & accuracy graph
│     ├── plant_cancer_yesno.h5
│     └── accuracy_graph.png
│
└── data/
      ├── train/
      │     ├── healthy/
      │     └── cancer/
      └── val/
            ├── healthy/
            └── cancer/
```

You **must** place your dataset in the correct folders before training.

---

# 🧪 Creating and Using the Virtual Environment (Windows)

## 1️⃣ Open a terminal (Command Prompt)
Press **Start → type "cmd" → Enter**

## 2️⃣ Navigate to your project folder

```cmd
cd C:\path\to\jadenScience
```

## 3️⃣ Create the virtual environment

```cmd
python -m venv venv
```

This creates a folder named `venv/`.

## 4️⃣ Activate the virtual environment

```cmd
venv\Scripts\activate
```

If successful, you will see:

```
(venv) C:\path\to\jadenScience>
```

## 5️⃣ Install dependencies

```cmd
pip install tensorflow pillow matplotlib
```

---

# 🧠 Training the Model

Once your virtual environment is activated and dependencies installed, run:

```cmd
python train.py
```

This will:

- Load images from `data/train` and `data/val`
- Train the MobileNetV2 model
- Save the trained model to:

```
model/plant_cancer_yesno.h5
```

- Generate an accuracy graph:

```
model/accuracy_graph.png
```

---

# 🌼 Running the GUI Predictor

After training, run:

```cmd
python gui_predict.py
```

A small window will appear with:

- A **Choose Image** button  
- A label showing prediction results  

The prediction will display as:

```
CANCER (92.15%)
```

or

```
HEALTHY (87.03%)
```

---

# 📸 Dataset Requirements

Your dataset **must** be arranged like this:

```
data/
  train/
    healthy/
    cancer/
  val/
    healthy/
    cancer/
```

Recommended minimum images:

| Folder | Minimum | Good | Best |
|--------|---------|------|-------|
| train/healthy | 20 | 100 | 300+ |
| train/cancer | 20 | 100 | 300+ |
| val/healthy | 5 | 20 | 50+ |
| val/cancer | 5 | 20 | 50+ |

More images = better accuracy.

---

# 🧬 Model Details

- **Architecture:** MobileNetV2  
- **Training Strategy:** Transfer Learning + Fine Tuning  
- **Input Size:** 224×224 RGB  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Output Classes:** `["cancer", "healthy"]`  

---

# 🧾 Science Fair Explanation (Simple)

> We trained an AI model to recognize whether a plant leaf is healthy or shows cancer signs.  
> The model learns by analyzing many example images.  
> After training, it can predict new images with high confidence.  
> This demonstrates how machine learning can help farmers detect plant diseases earlier.

---

# ⭐ If You Get Errors

Make sure:

1. Your virtual environment is activated  
2. TensorFlow is installed inside the venv  
3. A trained model exists in the `/model` folder  
4. You run scripts from the project root  

If you need help, open an issue or contact the project author.

---

# 🎉 Enjoy exploring AI and plant science!

