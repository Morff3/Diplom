import os
import sqlite3
import torch
import torchaudio
import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Конфигурация приложения Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Используйте свой секретный ключ

# База данных SQLite
DATABASE = 'users.db'

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
    return g.db

@app.teardown_appcontext
def close_db(error):
    if 'db' in g:
        g.db.close()

# Создание таблицы пользователей, если она не существует
def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        phone TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
        db.commit()

init_db()

# Загрузка модели классификации речи
classification_model = tf.keras.models.load_model("speech_classification_model.h5")

# Параметры модели генерации речи
language = 'ru'
model_id = 'v4_ru'
sample_rate = 48000
speaker = 'xenia'
device = torch.device('cpu')

# Загрузка модели генерации речи
generation_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
generation_model.to(device)

# Проверка, авторизован ли пользователь
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

# Главная страница с формой авторизации
@app.route('/')
def login():
    return render_template('login.html')

# Обработка авторизации
@app.route('/login', methods=['POST'])
def login_post():
    phone = request.form.get('phone')
    password = request.form.get('password')

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE phone = ?', (phone,)).fetchone()

    if user is None or not check_password_hash(user[4], password):
        flash('Неправильный номер телефона или пароль')
        return redirect(url_for('login'))

    session['user_id'] = user[0]  # Устанавливаем сессию пользователя
    return redirect(url_for('classification'))

# Страница регистрации
@app.route('/registration')
def registration():
    return render_template('registration.html')

# Обработка регистрации
@app.route('/register', methods=['POST'])
def register():
    first_name = request.form.get('firstName')
    last_name = request.form.get('lastName')
    phone = request.form.get('phone')
    password = request.form.get('password')

    db = get_db()
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    try:
        db.execute('INSERT INTO users (first_name, last_name, phone, password) VALUES (?, ?, ?, ?)',
                   (first_name, last_name, phone, hashed_password))
        db.commit()
    except sqlite3.IntegrityError:
        flash('Этот номер телефона уже зарегистрирован')
        return redirect(url_for('registration'))

    flash('Регистрация успешна, теперь войдите в систему')
    return redirect(url_for('login'))

# Классификация речи (требуется авторизация)
@app.route('/classification')
@login_required
def classification():
    return render_template('classification.html')

# Обработка классификации аудио
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and file.filename.endswith('.wav'):
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Извлечение MFCC и предсказание
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_mean = np.expand_dims(mfcc_mean, axis=0)

        prediction = classification_model.predict(mfcc_mean)
        predicted_label = np.round(prediction).astype(int)

        if predicted_label == 0:
            result = "Настоящая речь"
        else:
            result = "Синтетическая речь"

        return render_template('classification.html', result=result, audio_file=filename)

    return redirect(url_for('classification'))

# Маршрут для получения загруженных аудиофайлов
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Страница генерации речи (требуется авторизация)
@app.route('/generation')
@login_required
def generation():
    return render_template('generation.html')

# Обработка генерации речи
@app.route('/generate_audio', methods=['POST'])
@login_required
def generate_audio():
    text = request.form.get('text')

    if not text:
        return redirect(url_for('generation'))

    # Генерация аудио
    audio = generation_model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
    file_name = "synthetic_speech.wav"
    file_path = os.path.join('generated_audio', file_name)

    # Сохранение файла
    torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)

    # Возвращаем JSON с URL сгенерированного файла
    return {'audio_url': url_for('download_audio', filename=file_name)}

# Маршрут для скачивания сгенерированных аудиофайлов
@app.route('/generated_audio/<filename>')
@login_required
def download_audio(filename):
    return send_from_directory('generated_audio', filename)

# Выход из системы
@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))

# Запуск приложения
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('generated_audio'):
        os.makedirs('generated_audio')
    app.run(debug=True)
