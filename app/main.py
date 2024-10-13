from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi import Request
import nltk

# Устанавливаем путь к данным NLTK внутри проекта
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Обязательно скачать ресурсы nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = FastAPI()

# Подключение статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Загрузка модели эмбеддингов
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Загрузка предварительно обученных данных
with open('models/data_documents.pkl', 'rb') as file:
    documents = pickle.load(file)

with open('models/data_embeddings.pkl', 'rb') as file:
    embeddings = pickle.load(file)


# Функция для предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Классификация нового документа
def classify_new_document(text: str):
    preprocessed_text = preprocess_text(text)
    new_embedding = model.encode([preprocessed_text])

    similarities = cosine_similarity(new_embedding, embeddings)
    most_similar_index = np.argmax(similarities)
    predicted_cluster = documents[most_similar_index]['cluster']

    return predicted_cluster


# Главная страница (HTML)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Загрузка и классификация документа
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    # Преобразуем байты в строку
    text = content.decode('utf-8')

    # Извлечение текста после "Text: " (если есть)
    start_index = text.find("Text: ")
    if start_index != -1:
        extracted_text = text[start_index + 6:].strip()
    else:
        extracted_text = text

    # Классификация текста
    cluster = classify_new_document(extracted_text)

    # Преобразуем numpy.int32 в обычный int
    return {"cluster": int(cluster)}