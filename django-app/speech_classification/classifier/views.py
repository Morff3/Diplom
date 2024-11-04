# views.py

import os
import uuid
import numpy as np
import tensorflow as tf
import librosa
import torch
import torchaudio
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

# Загрузка модели классификации речи
CLASSIFICATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'speech_classification_model.h5')
classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)


# Функция для извлечения признаков MFCC
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


# Представление для регистрации
def register_view(request):
    if request.method == 'POST':
        first_name = request.POST.get('firstName')
        last_name = request.POST.get('lastName')
        phone = request.POST.get('phone')
        password = request.POST.get('password')

        if User.objects.filter(username=phone).exists():
            return render(request, 'registration.html', {'error': 'Пользователь с таким номером уже зарегистрирован.'})

        user = User.objects.create_user(username=phone, password=password, first_name=first_name, last_name=last_name)
        user.save()
        login(request, user)
        return redirect('classify_speech')
    else:
        return render(request, 'registration.html')


# Представление для входа
def login_view(request):
    if request.method == 'POST':
        phone = request.POST.get('phone')
        password = request.POST.get('password')

        user = authenticate(request, username=phone, password=password)
        if user is not None:
            login(request, user)
            return redirect('classify_speech')
        else:
            return render(request, 'login.html', {'error': 'Неверный номер телефона или пароль.'})
    else:
        return render(request, 'login.html')


# Представление для выхода
def logout_view(request):
    logout(request)
    return redirect('login')


# Представление для класификации речи (требуется авторизация)
@login_required(login_url='login')
def classify_speech(request):
    result = None
    audio_url = None

    if request.method == 'POST' and request.FILES.get('file'):
        audio_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        uploaded_file_url = fs.url(filename)

        # Полный путь к загруженному файлу
        file_path = os.path.join(fs.location, filename)

        # Извлекаем MFCC и делаем предсказание
        mfcc = extract_mfcc(file_path)
        mfcc = np.expand_dims(mfcc, axis=0)
        prediction = classification_model.predict(mfcc)
        predicted_label = np.round(prediction)[0][0]

        if predicted_label == 0:
            result = "Настоящая речь"
        else:
            result = "Синтетическая речь"

        audio_url = uploaded_file_url

    return render(request, 'classify.html', {
        'result': result,
        'audio_url': audio_url
    })


# Загрузка модели Silero TTS
language = 'ru'
model_id = 'v4_ru'
sample_rate = 48000
speaker = 'xenia'
device = torch.device('cpu')

tts_model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts',
                              language=language, speaker=model_id)
tts_model.to(device)


# Представление для генерации речи (требуется авторизация)
@login_required(login_url='login')
def generation_view(request):
    return render(request, 'generation.html')


# Представление для генерации аудио (требуется авторизация)
@login_required(login_url='login')
@csrf_exempt
def generate_audio(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if not text:
            return JsonResponse({'error': 'Текст не может быть пустым'})

        try:
            # Генерация аудио
            audio = tts_model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)

            # Генерация уникального имени файла
            unique_filename = f"generated_{uuid.uuid4().hex}.wav"
            file_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
            torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)

            # URL для доступа к аудио
            audio_url = settings.MEDIA_URL + unique_filename
            return JsonResponse({'audio_url': audio_url})
        except Exception as e:
            return JsonResponse({'error': f'Ошибка генерации: {str(e)}'})
    else:
        return JsonResponse({'error': 'Неверный метод запроса'})
