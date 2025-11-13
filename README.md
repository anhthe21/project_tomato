# 🍅 ỨNG DỤNG CNN TRONG PHÂN LOẠI CÀ CHUA THEO ĐỘ CHÍN (XANH & CHÍN)

## 🌱 Giới thiệu
Đề tài này sử dụng **mạng nơ-ron tích chập (Convolutional Neural Network - CNN)** để **phân loại cà chua theo độ chín** thành hai loại:  
- 🟢 **Xanh (Unripe)**  
- 🔴 **Chín (Ripe)**  

Dự án được thực hiện trong khuôn khổ môn học **Trí tuệ nhân tạo (AI)** tại **Trường Đại học Công nghiệp Hà Nội (HAUI)**.  
Toàn bộ mã nguồn được viết bằng **Python** và **TensorFlow/Keras**.

---

## 🎯 Mục tiêu
- Hiểu quy trình xây dựng mô hình CNN từ đầu.  
- Huấn luyện mô hình phân loại ảnh cơ bản.  
- Ứng dụng mô hình CNN để nhận diện độ chín của cà chua.

---

## 🧠 Kiến thức sử dụng
- Lập trình **Python cơ bản**
- Thư viện **TensorFlow / Keras** (Deep Learning)
- **Xử lý ảnh** với OpenCV / Pillow
- **NumPy**, **Matplotlib** để hiển thị dữ liệu và kết quả

---

## ⚙️ Cấu trúc thư mục dự án

```bash
PROJECT_TOMATO/
│
├── data/
│   ├── train/       # Ảnh dùng để huấn luyện
│   ├── test/        # Ảnh dùng để kiểm thử
│   ├── val/         # Ảnh dùng để đánh giá
│
├── models/
│   └── best_model.h5       # Mô hình CNN đã huấn luyện
│
├── src/
│   ├── data_prep.py        # Tiền xử lý dữ liệu ảnh
│   ├── train.py            # Huấn luyện mô hình CNN
│   ├── evaluate.py         # Đánh giá mô hình
│   ├── infer.py            # Dự đoán ảnh mới
│
├── README.md               # File hướng dẫn này
├── .gitignore              # Chỉ định các file/thư mục Git bỏ qua
├── requirements.txt        # Danh sách các thư viện cần thiết
└── .gitattributes          
```

---

## 🚀 Cài đặt & Chạy bằng **uv**

### 1️⃣ Cài đặt công cụ `uv`
```bash
pip install uv
```

### 2️⃣ Tạo và kích hoạt môi trường ảo
```bash
uv venv
```

**Windows:**
```bash
.venv\Scripts\activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3️⃣ Cài đặt thư viện
Nếu bạn có file `pyproject.toml`:
```bash
uv sync
```

Hoặc cài thủ công:
```bash
uv pip install tensorflow numpy matplotlib pillow opencv-python scipy
```

---

## 🧩 Huấn luyện, đánh giá & dự đoán

### Huấn luyện mô hình
```bash
uv run src/train.py
```

### Đánh giá mô hình
```bash
uv run src/evaluate.py
```

### Dự đoán ảnh mới
```bash
uv run src/infer.py
```

---

## 🧬 Cấu trúc mô hình CNN (đề xuất)

|Lớp	                                | Mô tả                                        |
|Block 1                                | (Conv, BatchNorm, Pool, Dropout)	32 filters |
|Block 2                                | (Conv, BatchNorm, Pool, Dropout)	64 filters |
|Block 3                                | (Conv, BatchNorm, Pool, Dropout)	128 filters|
|Block 4                                | (Conv, BatchNorm, Pool, Dropout)	256 filters|
|Flatten	                            | Chuyển ma trận thành vector                  |
|Dense(512, ReLU, BatchNorm, Dropout)	| Lớp fully-connected                          |
|Dense(128, ReLU, BatchNorm, Dropout)	| Lớp fully-connected                          |
|Dense(2, Softmax)	                    | Phân loại 2 lớp (Xanh/Chín)                  |

---

## 📚 Tài liệu tham khảo
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)  
- [Keras API Documentation](https://keras.io/api/)  
- [Pillow – Python Imaging Library](https://pillow.readthedocs.io/en/stable/)  

---

## ❤️ Ghi chú
Dự án này chỉ phục vụ **mục đích học tập và nghiên cứu**, **không dùng cho mục đích thương mại**.  
Có thể mở rộng thêm để nhận diện nhiều mức độ chín khác nhau (xanh, vàng, chín, thối...).