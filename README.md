# 🍅 ỨNG DỤNG CNN TRONG PHÂN LOẠI CÀ CHUA THEO ĐỘ CHÍN

## 🌱 Giới thiệu
Đề tài này nghiên cứu và xây dựng mô hình **Mạng nơ-ron tích chập (Convolutional Neural Network - CNN)** để **phân loại cà chua theo độ chín** thành hai nhãn:
- 🟢 **Xanh (Unripe)**
- 🔴 **Chín (Ripe)**

Dự án được thực hiện trong khuôn khổ môn học **Trí tuệ nhân tạo (AI)** tại **Trường Đại học Công nghiệp Hà Nội (HAUI)**.  
Mã nguồn được phát triển bằng ngôn ngữ **Python** và thư viện **TensorFlow/Keras**.

![Training History](models/training_history.png)
*(Biểu đồ quá trình huấn luyện mô hình)*

---

## 🎯 Chức năng chính
1. **Huấn luyện:** Xây dựng mô hình CNN tự động lưu trọng số tốt nhất.
2. **Đánh giá:** Tính toán độ chính xác (Accuracy), vẽ Ma trận nhầm lẫn (Confusion Matrix).
3. **Dự đoán (Inference):** Nhận diện độ chín từ một file ảnh bất kỳ bên ngoài.

---

## ⚙️ Cấu trúc thư mục

```bash
PROJECT_TOMATO/
│
├── .venv/                   # Môi trường ảo Python
├── data/                    # Dữ liệu ảnh
│   ├── train/               # Ảnh dùng để huấn luyện (Data Augmentation)
│   ├── val/                 # Ảnh dùng để đánh giá quá trình train
│   └── test/                # Ảnh dùng để kiểm thử độc lập
│
├── models/                  # Nơi lưu trữ mô hình và log
│   ├── best_model.h5        # File trọng số tốt nhất đã train
│   ├── training_history.npy # File lịch sử huấn luyện (numpy)
│   └── training_history.png # Biểu đồ trực quan Accuracy/Loss
│
├── src/                     # Mã nguồn chính
│   ├── data_prep.ipynb      # Notebook xử lý/khám phá dữ liệu (Jupyter)
│   ├── train.py             # Script huấn luyện mô hình
│   ├── evaluate.py          # Script đánh giá & vẽ Confusion Matrix
│   └── infer.py             # Script dự đoán ảnh đơn lẻ (Ứng dụng thực tế)
│
├── README.md                # Tài liệu hướng dẫn
├── requirements.txt         # Danh sách thư viện
└── .gitattributes           # Cấu hình Git
🛠️ Cài đặt môi trường
Dự án sử dụng trình quản lý gói uv (hoặc pip chuẩn).

1️⃣ Cài đặt công cụ
Bash

pip install uv

2️⃣ Tạo môi trường & Cài thư viện
Bash

# Tạo môi trường ảo
uv venv

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate

# Cài đặt các thư viện cần thiết
uv pip install tensorflow numpy matplotlib pillow opencv-python scikit-learn seaborn

🚀 Hướng dẫn sử dụng
1. Huấn luyện mô hình (Training)
Quá trình này sẽ đọc ảnh từ data/train, huấn luyện 50 epochs và lưu model vào models/best_model.h5.

Bash

python src/train.py

# Hoặc dùng uv:
uv run src/train.py

2. Đánh giá mô hình (Evaluation)
Script này tính toán độ chính xác trên tập dữ liệu kiểm thử và vẽ biểu đồ nhầm lẫn để xem model hay sai ở đâu.

Bash

python src/evaluate.py

3. Dự đoán ảnh mới (Inference)
Dùng để kiểm tra một tấm ảnh cà chua bất kỳ (tải từ mạng hoặc chụp điện thoại).

Cách 1: Chạy chế độ tương tác (Kéo thả ảnh vào)

Bash

python src/infer.py

Cách 2: Chạy bằng dòng lệnh

Bash

python src/infer.py --image "data/test/ca_chua_xanh_01.jpg"

🧬 Kiến trúc mô hình CNN
Mô hình sử dụng kiến trúc tuần tự (Sequential) với 4 khối tích chập:
-----------------------------------------------------------------------------
| Block  |	Loại lớp (Layer)        |  Bộ lọc (Filters)   | Output Shape    |
| Input  |	    Image	            |       -	          | (128, 128, 3)   |
|   1	 |  Conv2D + BN + MaxPool   |       32	          | (64, 64, 32)    |
|   2	 |  Conv2D + BN + MaxPool	|       64	          | (32, 32, 64)    |
|   3	 |  Conv2D + BN + MaxPool	|       128	          | (16, 16, 128)   |
|   4	 |  Conv2D + BN + MaxPool	|       256	          | (8, 8, 256)     |
|   FC	 |  Flatten + Dense	        |    64 neurons	      | Vector          |
|   FC	 |      Dense	            |    128 neurons	  | Vector          |
| Output |  Dense (Softmax)         |    2 neurons	      | (Xanh, Chín)    |
-----------------------------------------------------------------------------

📊 Kết quả thực nghiệm
(Số liệu ví dụ dựa trên lần huấn luyện gần nhất)
- Training Accuracy: ~90%
- Validation Accuracy: ~88%
- Loss: Hội tụ tốt sau khoảng 20 epochs.