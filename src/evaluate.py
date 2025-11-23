"""
evaluate.py - Đánh giá hiệu suất mô hình (Accuracy & Confusion Matrix)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Fix Unicode cho Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ========== CẤU HÌNH ==========
IMG_SIZE = (128, 128)
MODEL_PATH = 'models/best_model.h5'
TEST_DIR = 'data/val'  # Dùng tập validation hoặc test

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        return None
    print(f"Đang tải mô hình từ {MODEL_PATH}...")
    return tf.keras.models.load_model(MODEL_PATH)

def evaluate_dataset(model):
    print("\n" + "="*60)
    print("ĐÁNH GIÁ TRÊN TẬP DỮ LIỆU")
    print("="*60)

    # Chuẩn bị dữ liệu (chỉ rescale, KHÔNG augment)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    if not os.path.exists(TEST_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {TEST_DIR}")
        return

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False # QUAN TRỌNG: Không trộn để khớp nhãn
    )

    # Lấy tên các lớp
    class_names = list(test_generator.class_indices.keys())
    print(f"Các lớp: {class_names}")

    # Dự đoán
    print("Đang thực hiện dự đoán...")
    Y_pred = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1) # Nhãn dự đoán
    y_true = test_generator.classes    # Nhãn thực tế

    # 1. Ma trận nhầm lẫn (Confusion Matrix)
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.tight_layout()
    plt.show()

    # 2. Báo cáo chi tiết
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 3. Hiển thị ảnh sai (nếu có)
    errors = np.where(y_pred != y_true)[0]
    if len(errors) > 0:
        print(f"\nTìm thấy {len(errors)} ảnh dự đoán sai. Hiển thị tối đa 5 ảnh:")
        plt.figure(figsize=(15, 5))
        for i, error_idx in enumerate(errors[:5]):
            img_path = os.path.join(TEST_DIR, test_generator.filenames[error_idx])
            img = plt.imread(img_path)
            
            true_label = class_names[y_true[error_idx]]
            pred_label = class_names[y_pred[error_idx]]
            confidence = np.max(Y_pred[error_idx])
            
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", color='red', fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("\nTuyệt vời! Mô hình dự đoán đúng 100%.")

# ========== MAIN ==========
if __name__ == "__main__":
    model = load_trained_model()
    if model:
        evaluate_dataset(model)
        print("\nĐã hoàn tất đánh giá.")