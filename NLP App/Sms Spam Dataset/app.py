from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  


class TextData(BaseModel):
    text: str


app = FastAPI()
templates = Jinja2Templates(directory="templates")

ps = PorterStemmer()

try:
    stopwords.words("english")
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")


def transform_text(text):
    text = re.sub(r"\W", " ", str(text))

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and len(i) > 2:
            y.append(ps.stem(i))
    return " ".join(y)
try:
    pipeline = joblib.load("spam_classifier_model.pkl")
except FileNotFoundError:
    pipeline = None
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    pipeline = None


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, data: TextData):

    if pipeline is None:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "prediction": "HATA: Model sunucuda yüklenemedi."},
        )

    transformed_input = transform_text(data.text)

    prediction_raw = pipeline.predict([transformed_input])

    prediction_label = "Spam" if prediction_raw[0] == 1 else "Ham"

    return templates.TemplateResponse(
        "result.html", {"request": request, "prediction": prediction_label}
    )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
