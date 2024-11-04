# Импорт необходимых библиотек
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Функция для извлечения признаков MFCC из аудиофайла
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


# Путь к папке с аудиофайлами (нужно настроить под ваши данные)
REAL_VOICE_PATH = "real_voice"  # папка с настоящими записями
FAKE_VOICE_PATH = "fake_voice"  # папка с синтетическими записями

# Сбор данных: создаем массивы признаков и меток
X = []  # признаки (MFCC)
y = []  # метки (0 - настоящая речь, 1 - синтетическая речь)


# Функция для загрузки и извлечения признаков из аудиофайлов
def load_data(voice_path, label):
    for file_name in os.listdir(voice_path):
        file_path = os.path.join(voice_path, file_name)
        if file_path.endswith(".wav"):  # только .wav файлы
            mfcc = extract_mfcc(file_path)
            X.append(mfcc)
            y.append(label)


# Загружаем настоящие записи (метка 0) и синтетические записи (метка 1)
load_data(REAL_VOICE_PATH, 0)
load_data(FAKE_VOICE_PATH, 1)

# Преобразование данных в формат numpy
X = np.array(X)
y = np.array(y)

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Создание модели нейронной сети
def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # для бинарной классификации
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Создаем модель
input_shape = (X_train.shape[1],)
model = create_model(input_shape)

# Обучение модели
history = model.fit(X_train, y_train, epochs=30, validation_split=0.2)

# Оценка модели на тестовых данных
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)  # Округляем предсказания до ближайшего целого (0 или 1)

# Печатаем отчёт о классификации
print(classification_report(y_test, y_pred_rounded))

# Сохранение обученной модели
model.save("speech_classification_model.h5")


# Загрузка модели для предсказаний
def load_trained_model():
    return tf.keras.models.load_model("speech_classification_model.h5")


# Функция для предсказания на основе нового аудиофайла
def predict_voice(file_path, model):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # Преобразуем в форму (1, 13) для подачи в модель
    prediction = model.predict(mfcc)
    predicted_label = np.round(prediction)  # Округляем предсказание до 0 или 1
    if predicted_label == 0:
        print(f"Файл {file_path} классифицирован как настоящая речь.")
    else:
        print(f"Файл {file_path} классифицирован как синтетическая речь.")
