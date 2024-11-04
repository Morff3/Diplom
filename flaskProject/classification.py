import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

# Конфигурация приложения Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'static/audio/'
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Загрузка обученной модели
model = tf.keras.models.load_model("speech_classification_model.h5")


# Проверка допустимого расширения файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Функция для извлечения MFCC признаков из аудиофайла
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


# Главная страница для классификации
@app.route('/')
def index():
    return render_template('classification.html')


# Обработка загрузки и классификации файла
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Извлекаем MFCC и делаем предсказание
        mfcc = extract_mfcc(file_path)
        mfcc = np.expand_dims(mfcc, axis=0)  # Преобразуем в форму (1, 13) для подачи в модель
        prediction = model.predict(mfcc)
        predicted_label = np.round(prediction)

        # Копируем файл в папку static/audio для прослушивания
        audio_file_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        os.rename(file_path, audio_file_path)

        # Определяем результат классификации
        if predicted_label == 0:
            result = "Настоящая речь"
        else:
            result = "Синтетическая речь"

        return render_template('classification.html', result=result, audio_file=filename)

    return redirect(url_for('index'))


# Возвращает файл для прослушивания
@app.route('/static/audio/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)


# Навигация на страницу генерации
@app.route('/generation')
def go_to_generation():
    return redirect("http://localhost:5001/generation")


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['AUDIO_FOLDER']):
        os.makedirs(app.config['AUDIO_FOLDER'])
    app.run(debug=True, port=5000)
