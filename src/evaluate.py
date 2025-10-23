# src/evaluate.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "C:\\Users\\scubi\\Documents\\Projects\\project_tomato\\models\\best_model.h5"
DATA_DIR = "C:\\Users\\scubi\\Documents\\Projects\\project_tomato\\data"
IMG_SIZE = (128,128)
BATCH_SIZE = 32

model = load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

preds = model.predict(test_gen, verbose=1)
y_pred = (preds.ravel() > 0.5).astype(int)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=labels))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))