import customtkinter as ctk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Tải model
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
model = load_model('best_deep_modelSE.h5')
detector = MTCNN()

def detect_and_predict_emotion(img):
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return img, None, None

    for face in faces:
        x, y, w, h = face['box']
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = prediction[0][emotion_index]

        label = f"{emotion} ({confidence*100:.1f}%)"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return img, emotion, confidence
    return img, None, None

def open_image():
    file_path = fd.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            result_label.configure(text="Không thể mở ảnh.")
            return

        result_img, emotion, confidence = detect_and_predict_emotion(img)
        rgb_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).resize((400, 400))
        tk_img = ImageTk.PhotoImage(pil_img)
        image_label.configure(image=tk_img)
        image_label.image = tk_img

        if emotion:
            result_label.configure(text=f"Cảm xúc: {emotion} ({confidence*100:.1f}%)")
        else:
            result_label.configure(text="Không phát hiện khuôn mặt.")

# Khởi tạo giao diện
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue")  

app = ctk.CTk()
app.title("Nhận diện cảm xúc")
app.geometry("500x650")

title = ctk.CTkLabel(app, text="Nhận diện cảm xúc từ ảnh", font=ctk.CTkFont(size=20, weight="bold"))
title.pack(pady=20)

select_button = ctk.CTkButton(app, text="Chọn ảnh", command=open_image, width=200, height=40, corner_radius=8)
select_button.pack(pady=10)

image_label = ctk.CTkLabel(app, text="")
image_label.pack(pady=20)

result_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=16))
result_label.pack(pady=10)

app.mainloop()
