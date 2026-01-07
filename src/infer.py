"""
infer.py - Dự đoán độ chín cà chua từ ảnh bất kỳ
Sử dụng: python src/infer.py --image "duong/dan/toi/anh.jpg"
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image

# Fix lỗi hiển thị tiếng Việt trên Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ========== CẤU HÌNH ==========
MODEL_PATH = 'models/best_model.h5'
IMG_SIZE = (128, 128)

# Nhãn (Cần khớp với thứ tự thư mục lúc train: chin -> xanh)
# Mặc định flow_from_directory sắp xếp alphabe: 'chin' (0), 'xanh' (1)
CLASS_NAMES = {
    0: 'Cà chua CHÍN (Ripe)',
    1: 'Cà chua XANH (Unripe)'
}

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f" Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        print("   Vui lòng chạy 'python src/train.py' trước.")
        sys.exit(1)
    
    print(f" Đang tải mô hình từ {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(" Tải mô hình thành công!")
        return model
    except Exception as e:
        print(f" Lỗi khi tải mô hình: {e}")
        sys.exit(1)

def predict_image(model, img_path):
    if not os.path.exists(img_path):
        print(f" Lỗi: Không tìm thấy ảnh tại {img_path}")
        return

    print(f"\n🔍 Đang xử lý ảnh: {img_path}")
    
    try:
        # 1. Load ảnh và resize về đúng kích thước model yêu cầu
        img = image.load_img(img_path, target_size=IMG_SIZE)
        
        # 2. Chuyển sang mảng numpy và chuẩn hóa màu (0-255 -> 0-1)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch: (1, 128, 128, 3)
        img_array /= 255.0 

        # 3. Dự đoán
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0]) # Chuyển về xác suất
        
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        result_text = CLASS_NAMES.get(class_idx, "Không xác định")

        # 4. In kết quả ra màn hình
        print("-" * 40)
        print(f"   KẾT QUẢ: {result_text}")
        print(f"   Độ tin cậy: {confidence*100:.2f}%")
        print("-" * 40)

        # 5. Hiển thị ảnh kèm kết quả
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{result_text}\n({confidence*100:.2f}%)", color='red', fontsize=14)
        plt.show()

    except Exception as e:
        print(f" Lỗi khi dự đoán: {e}")

# ========== MAIN ==========
if __name__ == "__main__":
    # Tạo bộ đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Dự đoán độ chín cà chua")
    parser.add_argument('--image', type=str, help='Đường dẫn đến file ảnh cần dự đoán')
    
    args = parser.parse_args()

    # Load model 1 lần duy nhất
    model = load_model()

    # Nếu người dùng nhập tham số --image
    if args.image:
        predict_image(model, args.image)
    else:
        # Chế độ nhập tay liên tục nếu không truyền tham số
        print("\n Mẹo: Bạn có thể kéo thả file ảnh vào cửa sổ này để lấy đường dẫn.")
        while True:
            img_path = input("\n Nhập đường dẫn ảnh (hoặc gõ 'exit' để thoát): ").strip()
            
            # Xử lý xóa dấu ngoặc kép nếu có (do Windows copy path)
            if img_path.startswith('"') and img_path.endswith('"'):
                img_path = img_path[1:-1]
            
            if img_path.lower() in ['exit', 'quit', 'q']:
                print("Tạm biệt! ")
                break
            
            if img_path:
                predict_image(model, img_path)