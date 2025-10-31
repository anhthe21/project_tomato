"""
train.py - Huan luyen mo hinh CNN phan loai ca chua (Xanh/Chin)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Fix Unicode encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========== CAU HINH ==========
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Duong dan du lieu (dung forward slash hoat dong tren ca Windows/Linux)
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
MODEL_SAVE_PATH = 'models/best_model.h5'

# Tao thu muc models neu chua co
os.makedirs('models', exist_ok=True)

print("="*60)
print("BAT DAU HUAN LUYEN MO HINH CNN PHAN LOAI CA CHUA")
print("="*60)
print(f"Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Kich thuoc anh: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"So epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*60)


# ========== CHUAN BI DU LIEU ==========
print("\nDang tai du lieu...")

# Data Augmentation cho tap train (tang cuong du lieu)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Chi chuan hoa cho tap validation (khong augment)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load du lieu tu thu muc
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# So luong class
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"Tai du lieu thanh cong!")
print(f"   So luong anh train: {train_generator.samples}")
print(f"   So luong anh validation: {val_generator.samples}")
print(f"   Cac lop: {class_names}")
print(f"   So lop: {num_classes}")


# ========== XAY DUNG MO HINH CNN ==========
print("\nDang xay dung mo hinh CNN...")

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output Layer
    layers.Dense(num_classes, activation='softmax')
])

print("Mo hinh CNN da duoc xay dung!")
model.summary()


# ========== COMPILE MO HINH ==========
print("\nDang compile mo hinh...")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Compile hoan tat!")


# ========== CALLBACKS ==========
print("\nThiet lap callbacks...")

# 1. Early Stopping - Dung som neu khong cai thien
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 2. Model Checkpoint - Luu mo hinh tot nhat
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 3. Reduce Learning Rate - Giam LR khi plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

print("Callbacks da san sang!")


# ========== HUAN LUYEN MO HINH ==========
print("\n" + "="*60)
print("BAT DAU HUAN LUYEN MO HINH")
print("="*60 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("HUAN LUYEN HOAN TAT!")
print("="*60)


# ========== LUU KET QUA ==========
print("\nDang luu ket qua...")

# Luu lich su huan luyen
np.save('models/training_history.npy', history.history)

# Danh gia tren tap validation
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\nKET QUA CUOI CUNG:")
print(f"   Validation Loss: {val_loss:.4f}")
print(f"   Validation Accuracy: {val_acc*100:.2f}%")


# ========== VE DO THI ==========
print("\nDang tao do thi...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Do thi Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Do thi Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
print("Do thi da duoc luu tai: models/training_history.png")

plt.show()

# ========== TONG KET ==========
print("\n" + "="*60)
print("HOAN THANH TAT CA!")
print("="*60)
print(f"Mo hinh da luu tai: {MODEL_SAVE_PATH}")
print(f"Do thi da luu tai: models/training_history.png")
print(f"Lich su huan luyen: models/training_history.npy")
print("="*60)