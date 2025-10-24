# 🍅 ỨNG DỤNG CNN TRONG PHÂN LOẠI CÀ CHUA THEO ĐỘ CHÍN (XANH & CHÍN)

## 🌱 Giới thiệu
Đề tài này sử dụng mạng nơ-ron tích chập (Convolutional Neural Network - CNN) để **phân loại cà chua theo độ chín** thành hai loại: **Xanh (Unripe)** và **Chín (Ripe)**.

Đây là một dự án học tập trong môn **Trí tuệ nhân tạo (AI)**, được thực hiện tại **Trường Đại học Công nghiệp Hà Nội (HAUI)**.  
Toàn bộ mã nguồn được viết bằng **Python** và **TensorFlow/Keras**.

---

## 🎯 Mục tiêu
- Hiểu quy trình xây dựng mô hình CNN từ đầu.
- Huấn luyện mô hình phân loại ảnh đơn giản.
- Ứng dụng thực tế vào việc nhận diện độ chín của cà chua.

---

## 🧠 Kiến thức sử dụng
- Python cơ bản
- TensorFlow / Keras (Deep Learning)
- Xử lý ảnh (OpenCV, Pillow)
- NumPy, Matplotlib (hiển thị dữ liệu và kết quả)

---

## ⚙️ Cấu trúc thư mục dự án

```bash
PROJECT_TOMATO/
│
├── data/
│   ├── train/       # Ảnh dùng để huấn luyện (train)
│   ├── test/        # Ảnh dùng để kiểm thử (test)
│   ├── val/         # Ảnh dùng để đánh giá (validation)
│
├── models/
│   └── best_model.h5    # Mô hình CNN đã huấn luyện
│
├── notebooks/            # Notebook demo/training (tuỳ chọn)
│
├── src/
│   ├── data_prep.py      # Tiền xử lý dữ liệu ảnh
│   ├── train.py          # Huấn luyện mô hình CNN
│   ├── evaluate.py       # Đánh giá mô hình trên tập test
│   ├── infer.py          # Dự đoán ảnh mới
│
├── README.md
├── requirements.txt
└── .gitattributes
```

---

## 🚀 Cách cài đặt và chạy

### 1️⃣ Cài môi trường
Tạo môi trường ảo (tuỳ chọn):
```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows
# hoặc
source .venv/bin/activate         # Linux/Mac
```

Cài thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2️⃣ Huấn luyện mô hình
```bash
python src/train.py
```

### 3️⃣ Đánh giá mô hình
```bash
python src/evaluate.py
```

### 4️⃣ Dự đoán ảnh mới
```bash
python src/infer.py
```

---

## 🧩 Cấu trúc mô hình CNN (đề xuất)
- **Conv2D(32, 3x3, ReLU)**  
- **MaxPooling2D(2x2)**  
- **Conv2D(64, 3x3, ReLU)**  
- **MaxPooling2D(2x2)**  
- **Flatten**  
- **Dense(128, ReLU)**  
- **Dense(2, Softmax)**  

---

## 🧾 Tài liệu tham khảo
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [Keras API Documentation](https://keras.io/api/)
- [Python Image Library (Pillow)](https://pillow.readthedocs.io/en/stable/)

---

## ❤️ Ghi chú
Dự án chỉ phục vụ mục đích **học tập và nghiên cứu**, không được dùng cho mục đích thương mại.
