# src/infer.py
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "C:\\Users\\scubi\\Documents\\Projects\\project_tomato\\models\\best_model.h5"
IMG_SIZE = (224,224)

model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    p = model.predict(x)[0][0]
    label = "ripe" if p > 0.5 else "green"
    print(f"Prob ripe: {p:.4f} → Pred: {label}")

if __name__ == "__main__":
    img_path = sys.argv[1]
    predict_image(img_path)