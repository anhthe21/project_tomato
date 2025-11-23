"""
evaluate.py - Đánh giá mô hình CNN phân loại cà chua và kiểm thử thực tế
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing import image

# Fix Unicode cho Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ========== CẤU HÌNH ==========
IMG_SIZE = (128, 128)  # Phải khớp với train.py
MODEL_PATH = 'models/best_model.h5'
TEST_DIR = 'data/val'  # Sử dụng tập validation để đánh giá (hoặc đổi thành 'data/test' nếu có)

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

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False # QUAN TRỌNG: Không trộn để khớp nhãn với dự đoán
    )

    # Lấy tên các lớp
    class_names = list(test_generator.class_indices.keys())
    print(f"Các lớp: {class_names}")

    # Dự đoán toàn bộ tập dữ liệu
    print("Đang thực hiện dự đoán...")
    Y_pred = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1) # Lấy nhãn dự đoán cao nhất
    y_true = test_generator.classes    # Lấy nhãn thực tế

    # 1. Ma trận nhầm lẫn (Confusion Matrix)
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
    plt.ylabel('Nhãn thực tế (True Label)')
    plt.xlabel('Nhãn dự đoán (Predicted Label)')
    plt.tight_layout()
    plt.show()

    # 2. Báo cáo chi tiết (Classification Report)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 3. Hiển thị một số ảnh dự đoán sai (nếu có)
    errors = np.where(y_pred != y_true)[0]
    if len(errors) > 0:
        print(f"\nTìm thấy {len(errors)} ảnh dự đoán sai. Hiển thị 5 ảnh đầu tiên:")
        plt.figure(figsize=(15, 5))
        for i, error_idx in enumerate(errors[:5]):
            img_path = os.path.join(TEST_DIR, test_generator.filenames[error_idx])
            img = plt.imread(img_path)
            
            true_label = class_names[y_true[error_idx]]
            pred_label = class_names[y_pred[error_idx]]
            confidence = np.max(Y_pred[error_idx])
            
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", color='red')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("\nTuyệt vời! Mô hình dự đoán đúng 100% trên tập dữ liệu này.")

    return class_names

def predict_single_image(model, img_path, class_names):
    """Hàm dự đoán cho một file ảnh bất kỳ"""
    print("\n" + "="*60)
    print(f"DỰ ĐOÁN ẢNH LẺ: {img_path}")
    print("="*60)

    try:
        # Load và tiền xử lý ảnh
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch (1, 128, 128, 3)
        img_array /= 255.0 # Chuẩn hóa giống lúc train

        # Dự đoán
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Dùng softmax để lấy xác suất chuẩn
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = np.max(predictions[0])

        # Hiển thị kết quả
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Dự đoán: {predicted_class}\nĐộ tin cậy: {confidence*100:.2f}%")
        plt.axis('off')
        plt.show()

        print(f"--> Kết quả: {predicted_class}")
        print(f"--> Độ tin cậy: {confidence*100:.2f}%")
        
        return predicted_class

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")

# ========== MAIN ==========
if __name__ == "__main__":
    model = load_trained_model()
    
    if model:
        # 1. Đánh giá toàn bộ tập dữ liệu (Validation hoặc Test)
        class_names = evaluate_dataset(model)

        # 2. Ví dụ cách dự đoán 1 ảnh cụ thể (Bỏ comment để chạy thử)
        # Thay đường dẫn bên dưới bằng đường dẫn ảnh thực tế bạn muốn test
        # image_path_to_test = "test_images/ca_chua_xanh.jpg"
        # if os.path.exists(image_path_to_test):
        #     predict_single_image(model, image_path_to_test, class_names)
        
        print("\nĐã hoàn tất đánh giá.")