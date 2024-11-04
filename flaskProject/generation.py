import torch
import torchaudio
import os
from flask import Flask, render_template, request, jsonify, send_file, redirect

app = Flask(__name__)

# Параметры модели
language = 'ru'
model_id = 'v4_ru'
sample_rate = 48000
speaker = 'xenia'
device = torch.device('cpu')

# Загрузка модели генерации речи
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
model.to(device)

# Папка для хранения сгенерированных файлов
OUTPUT_FOLDER = 'generated_audio'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Главная страница для генерации речи
@app.route('/generation')
def generation():
    return render_template('generation.html')

# Обработка генерации аудиофайла
@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    text = request.form.get('text')

    if not text:
        return jsonify({'error': 'Текст для генерации не предоставлен'}), 400

    # Генерация аудио с помощью модели
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
    file_name = "synthetic_speech.wav"
    file_path = os.path.join(OUTPUT_FOLDER, file_name)

    # Сохранение аудиофайла
    torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)

    return jsonify({'audio_url': f'/download/{file_name}'})

# Маршрут для скачивания сгенерированного файла
@app.route('/download/<filename>', methods=['GET'])
def download_audio(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'Файл не найден'}), 404


# Навигация на страницу классификации
@app.route('/classification')
def go_to_classification():
    return redirect("http://localhost:5000")


if __name__ == '__main__':
    app.run(debug=True, port=5001)
