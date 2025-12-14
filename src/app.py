import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# --- CẤU HÌNH (Giống infer.py) ---
MODEL_PATH = 'models/best_model.h5'
IMG_SIZE = (128, 128)

# Nhãn: 0 -> Chin, 1 -> Xanh (Dựa theo code infer.py của bạn)
CLASS_NAMES = {
    0: 'CHÍN',
    1: 'XANH'
}

# --- KHỞI TẠO SERVER FLASK ---
app = Flask(__name__)
CORS(app) # Cho phép trang web gọi vào server này

# Load model khi server khởi động
print(f" Đang tải mô hình từ {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Tải mô hình thành công!")
except Exception as e:
    print(f" Lỗi: Không thể tải mô hình. Chi tiết: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được tải'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Không có file ảnh được gửi lên'}), 400

    file = request.files['file']
    
    try:
        # 1. Đọc file ảnh từ bộ nhớ (không cần lưu ra đĩa)
        img = Image.open(file.stream)
        
        # Đảm bảo ảnh ở hệ màu RGB (tránh lỗi nếu upload ảnh PNG trong suốt)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 2. Resize ảnh giống hệt infer.py
        img = img.resize(IMG_SIZE)

        # 3. Tiền xử lý (Chuyển sang array & Chuẩn hóa 0-1)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # 4. Dự đoán
        predictions = model.predict(img_array, verbose=0)
        
        # Xử lý kết quả (Softmax nếu model output nhiều lớp, hoặc sigmoid)
        # Giả sử model output dạng softmax như code cũ:
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        result_label = CLASS_NAMES.get(class_idx, "KHÔNG XÁC ĐỊNH")

        # Trả về JSON cho web
        return jsonify({
            'result': result_label,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy server ở cổng 5000
    app.run(host='0.0.0.0', port=5000, debug=True)