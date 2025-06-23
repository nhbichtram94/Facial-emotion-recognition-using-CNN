# Facial Emotion Recognition using CNN  
# Nhận Diện Cảm Xúc Gương Mặt bằng Mạng Nơ-ron Tích Chập (CNN)
This project implements a facial emotion recognition system using Convolutional Neural Networks (CNN) in Python.  
Dự án này xây dựng hệ thống nhận diện cảm xúc trên khuôn mặt sử dụng mạng nơ-ron tích chập CNN bằng ngôn ngữ Python.
# Emotions Recognized | Cảm xúc nhận diện:
- Angry (Giận dữ )  
- Disgust (Ghê tởm)  
- Fear (Sợ hãi)  
- Happy (Vui vẻ)  
- Sad (Buồn)  
- Surprise (Ngạc nhiên)  
- Neutral (Bình thường)
# Project Structure | Cấu trúc thư mục
NhanDienCamXucGuongMatCNN/
├── dataset/              # Dữ liệu huấn luyện và kiểm thử
├── model/                # File mô hình đã huấn luyện
├── haarcascade/          # Bộ nhận diện khuôn mặt (OpenCV)
├── train.py              # Huấn luyện mô hình
├── detect\_emotion.py     # Chạy nhận diện cảm xúc thời gian thực
├── test.py               # Kiểm thử mô hình
└── README.md             # Tệp hướng dẫn này
# How to Use | Cách sử dụng
1. Clone repo:
git clone https://github.com/nhbichtram94/Facial-emotion-recognition-using-CNN.git
cd Facial-emotion-recognition-using-CNN
2. Cài đặt thư viện:
pip install -r requirements.txt
3. Huấn luyện mô hình (nếu cần):
python train.py
4. Chạy nhận diện cảm xúc qua webcam:
python detect_emotion.py
# Mô hình & Công nghệ sử dụng
* CNN (Convolutional Neural Networks)
* OpenCV (face detection)
* TensorFlow / Keras
* Dataset: FER-2013 hoặc tương đương
* Input: ảnh khuôn mặt grayscale 48x48
# Yêu cầu hệ thống
* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy
* matplotlib
Cài tất cả bằng:
pip install -r requirements.txt
# Tác giả | Author
Nguyễn Hoàng Bích Trâm
GitHub: [@nhbichtram94](https://github.com/nhbichtram94)
# License
This project is licensed under the MIT License.
Dự án sử dụng giấy phép MIT – tự do sử dụng và chỉnh sửa có ghi nguồn.
1. Tạo file `README.md`  
2. Dán nội dung trên vào  
3. Lưu file và chạy:
git add README.md
git commit -m "Add clean and formatted README"
git push
