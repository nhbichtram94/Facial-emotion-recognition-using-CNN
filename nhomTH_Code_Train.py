
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Reshape, Multiply, ReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import regularizers
import tensorflow as tf

# ----------- Tham số ----------
NUM_CLASSES = 7
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 150


# ----------- Cosine Annealing ----------
def cosine_annealing(epoch, lr):
    T_max = EPOCHS
    eta_min = 1e-6
    eta_max = 1e-3
    return eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2

# ----------- Vẽ biểu đồ ----------
def plot_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.show()

# ----------- Load ảnh ----------
def load_data_from_folders(base_path):
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    label_to_index = {label: idx for idx, label in enumerate(emotion_labels)}
    images = []
    labels = []


    for emotion in emotion_labels:
        folder = os.path.join(base_path, emotion)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label_to_index[emotion])
    return np.array(images), np.array(labels)

# ----------- Chuẩn hóa và chia tập ----------
def prepare_data():
    train_path = r'C:\Users\lehoa\Downloads\emotion_dataset\train - Copy'
    test_path = r'C:\Users\lehoa\Downloads\emotion_dataset\test'

    train_images, train_labels = load_data_from_folders(train_path)
    test_images, test_labels = load_data_from_folders(test_path)

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    train_labels = to_categorical(train_labels, NUM_CLASSES)
    test_labels = to_categorical(test_labels, NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    return X_train, X_val, y_train, y_val, test_images, test_labels

# ----------- SE Block ----------
def se_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // reduction_ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    return Multiply()([input_tensor, se])

# ----------- CNN + SE ----------
def create_custom_cnn_se(input_shape=(48,48,1), num_classes=7, dropout_rate=0.3):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout_rate)(x)

    # Block 2
    x = Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout_rate)(x)

    # Block 3
    x = Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout_rate)(x)

    # Block 4
    x = Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout_rate)(x)

    # Block 5
    x = Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = Dropout(dropout_rate)(x)

    # Classification Head
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# ----------- Callbacks ----------
def get_callbacks(model_path):
    return [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        LearningRateScheduler(cosine_annealing, verbose=0)
    ]

# ----------- Data Generator (chỉ ImageDataGenerator) ----------
def data_generator_simple(x, y, batch_size):
    data_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    generator = data_gen.flow(x, y, batch_size=batch_size)
    while True:
        yield next(generator)

# ----------- Lưu kết quả ----------
def save_model_outputs(model, history, X_test, y_test):
    import pandas as pd
    import seaborn as sns

    # Lưu lịch sử huấn luyện
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)

    # Vẽ và lưu biểu đồ huấn luyện
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.close()

    # Lưu kiến trúc mô hình
    with open("model_architecture.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Load lại mô hình tốt nhất
    best_model = tf.keras.models.load_model('best_deep_modelSE.h5')

    # Dự đoán và đánh giá
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    report = classification_report(y_true, y_pred_classes, target_names=[
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    ])
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                yticklabels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("✅ Đã lưu toàn bộ kết quả.")

# ----------- Huấn luyện ----------
def train_model():
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data()

    train_gen = data_generator_simple(X_train, y_train, BATCH_SIZE)
    steps_per_epoch = len(X_train) // BATCH_SIZE

    model = create_custom_cnn_se()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=get_callbacks('best_deep_modelSE.h5')
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f'✅ Base Model - Val Accuracy: {val_acc:.2f} | Loss: {val_loss:.2f}')

    save_model_outputs(model, history, X_test, y_test)

    return model


if __name__ == '__main__':
    train_model()

