"""
train.py - Huấn luyện mô hình CNN phân loại cà chua (Xanh/Chín)
Tác giả: Thế Anh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========== CẤU HÌNH ==========
IMG_SIZE = (128, 128)  # Kích thước ảnh đầu vào
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Đường dẫn dữ liệu
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
MODEL_SAVE_PATH = 'models/best_model.h5'

# Tạo thư mục models nếu chưa có
os.makedirs('models', exist_ok=True)

print("="*60)
print("🍅 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH CNN PHÂN LOẠI CÀ CHUA")
print("="*60)
print(f"📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🖼️  Kích thước ảnh: {IMG_SIZE}")
print(f"📦 Batch size: {BATCH_SIZE}")
print(f"🔄 Số epochs: {EPOCHS}")
print(f"📈 Learning rate: {LEARNING_RATE}")
print("="*60)


# ========== CHUẨN BỊ DỮ LIỆU ==========
print("\n📂 Đang tải dữ liệu...")

# Data Augmentation cho tập train (tăng cường dữ liệu)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Chuẩn hóa pixel về [0,1]
    rotation_range=20,           # Xoay ngẫu nhiên ±20°
    width_shift_range=0.2,       # Dịch ngang 20%
    height_shift_range=0.2,      # Dịch dọc 20%
    shear_range=0.2,             # Biến dạng nghiêng
    zoom_range=0.2,              # Zoom in/out
    horizontal_flip=True,        # Lật ngang
    fill_mode='nearest'          # Điền pixel khi biến đổi
)

# Chỉ chuẩn hóa cho tập validation (không augment)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load dữ liệu từ thư mục
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',    # One-hot encoding
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Số lượng class
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"✅ Tải dữ liệu thành công!")
print(f"   🟢 Số lượng ảnh train: {train_generator.samples}")
print(f"   🔵 Số lượng ảnh validation: {val_generator.samples}")
print(f"   🏷️  Các lớp: {class_names}")
print(f"   📊 Số lớp: {num_classes}")


# ========== XÂY DỰNG MÔ HÌNH CNN ==========
print("\n🧠 Đang xây dựng mô hình CNN...")

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

print("✅ Mô hình CNN đã được xây dựng!")
model.summary()


# ========== COMPILE MÔ HÌNH ==========
print("\n⚙️  Đang compile mô hình...")

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Hàm loss cho multi-class
    metrics=['accuracy']
)

print("✅ Compile hoàn tất!")


# ========== CALLBACKS ==========
print("\n🔧 Thiết lập callbacks...")

# 1. Early Stopping - Dừng sớm nếu không cải thiện
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 2. Model Checkpoint - Lưu mô hình tốt nhất
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 3. Reduce Learning Rate - Giảm LR khi plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

print("✅ Callbacks đã sẵn sàng!")


# ========== HUẤN LUYỆN MÔ HÌNH ==========
print("\n" + "="*60)
print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
print("="*60 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("✅ HUẤN LUYỆN HOÀN TẤT!")
print("="*60)


# ========== LƯU KẾT QUẢ ==========
print("\n💾 Đang lưu kết quả...")

# Lưu lịch sử huấn luyện
np.save('models/training_history.npy', history.history)

# Đánh giá trên tập validation
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
print(f"   🔴 Validation Loss: {val_loss:.4f}")
print(f"   🟢 Validation Accuracy: {val_acc*100:.2f}%")


# ========== VẼ ĐỒ THỊ ==========
print("\n📈 Đang tạo đồ thị...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Đồ thị Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Đồ thị Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
print("✅ Đồ thị đã được lưu tại: models/training_history.png")

plt.show()

# ========== TỔNG KẾT ==========
print("\n" + "="*60)
print("🎉 HOÀN THÀNH TẤT CẢ!")
print("="*60)
print(f"📁 Mô hình đã lưu tại: {MODEL_SAVE_PATH}")
print(f"📊 Đồ thị đã lưu tại: models/training_history.png")
print(f"💾 Lịch sử huấn luyện: models/training_history.npy")
print("="*60)