from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer
from passlib.hash import bcrypt
from sqlalchemy import Column, Integer, String, create_engine
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, File, UploadFile

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.templating import Jinja2Templates
import jwt
import datetime
import os
import numpy as np
import torch
import torchaudio
import librosa
import tensorflow as tf
from pathlib import Path


# Настройки приложения FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Настройки для работы с базой данных SQLite
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Секретный ключ для JWT
SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"


# Модель пользователя для базы данных
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)


# Создание таблицы пользователей в базе данных
Base.metadata.create_all(bind=engine)


# Функции для взаимодействия с базой данных
def get_user_by_phone(db: Session, phone: str):
    return db.query(User).filter(User.phone == phone).first()


def create_user(db: Session, first_name: str, last_name: str, phone: str, password: str):
    user = User(
        first_name=first_name,
        last_name=last_name,
        phone=phone,
        password_hash=bcrypt.hash(password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# Зависимость для подключения к базе данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Кастомная зависимость для извлечения токена из cookie
async def get_current_user(access_token: str = Cookie(None), db: Session = Depends(get_db)):
    if access_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        user = get_user_by_phone(db, phone=payload.get("sub"))
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")


# Главная страница, перенаправляющая на страницу логина
@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/login")


# Страница регистрации
@app.get("/registration", response_class=HTMLResponse)
async def registration_page(request: Request):
    return templates.TemplateResponse("registration.html", {"request": request})


# Обработка данных регистрации
@app.post("/register", response_class=HTMLResponse)
async def register_user(
        request: Request,
        first_name: str = Form(...),
        last_name: str = Form(...),
        phone: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    if get_user_by_phone(db, phone):
        return templates.TemplateResponse("registration.html", {"request": request,
                                                                "error": "Пользователь с таким телефоном уже существует"})

    create_user(db, first_name, last_name, phone, password)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


# Страница авторизации
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# Обработка данных авторизации
@app.post("/login", response_class=HTMLResponse)
async def login_for_access_token(
        request: Request,
        phone: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    user = get_user_by_phone(db, phone=phone)
    if not user or not bcrypt.verify(password, user.password_hash):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Неверные учетные данные"})

    token_data = {
        "sub": user.phone,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    response = RedirectResponse(url="/classification", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=token, httponly=True)
    return response


# Функции и настройки для классификации и генерации речи
UPLOAD_FOLDER = "static/uploads"
GEN_AUDIO_PATH = "static/generated_audio"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(GEN_AUDIO_PATH).mkdir(parents=True, exist_ok=True)

# Загрузка модели классификации и настройки для модели TTS
classification_model = tf.keras.models.load_model("speech_classification_model.h5")
language = 'ru'
model_id = 'v4_ru'
sample_rate = 48000
speaker = 'xenia'
device = torch.device('cpu')
tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=language,
                              speaker=model_id)
tts_model.to(device)


def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


# Защищенный маршрут для классификации речи
@app.get("/classification", response_class=HTMLResponse)
async def classification_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
        request: Request, file: UploadFile = File(...), current_user: User = Depends(get_current_user)
):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    mfcc = extract_mfcc(file_location)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = classification_model.predict(mfcc)
    predicted_label = np.round(prediction)[0][0]
    result = "Настоящая речь" if predicted_label == 0 else "Синтетическая речь"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "audio_file": file.filename
    })


# Защищенный маршрут для генерации речи
@app.get("/generation", response_class=HTMLResponse)
async def generation_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("generation.html", {"request": request})


@app.post("/generate_audio", response_class=JSONResponse)
async def generate_audio(
    text: str = Form(...), current_user: User = Depends(get_current_user)
):
    if not text:
        return JSONResponse({"error": "Текст для генерации пуст"}, status_code=400)

    audio = tts_model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
    file_name = f"generated_{hash(text)}.wav"
    file_path = os.path.join(GEN_AUDIO_PATH, file_name)
    torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)
    return JSONResponse({"audio_url": f"/static/generated_audio/{file_name}"})

@app.get("/static/generated_audio/{filename}", response_class=FileResponse)
async def get_generated_audio(filename: str):
    file_path = os.path.join(GEN_AUDIO_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# Маршрут для выхода из системы
@app.get("/logout", response_class=RedirectResponse)
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response


# Маршрут для доступа к загруженным файлам
@app.get("/static/uploads/{filename}", response_class=FileResponse)
async def uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return FileResponse(file_path)
