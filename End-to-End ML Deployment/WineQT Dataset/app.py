from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("model_xgb.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]


class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, input_data: WineInput):
    # Ensure correct column order as in training
    feature_order = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    input_dict = input_data.model_dump()
    input_df = pd.DataFrame([input_dict])[feature_order]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Convert numpy types to Python native types
    prediction = int(prediction)
    prediction_proba = prediction_proba.astype(float)

    # Interpret the binary classification result
    if prediction == 0:
        quality_label = "Poor Quality"
        confidence = float(prediction_proba[0] * 100)
    else:
        quality_label = "Good Quality"
        confidence = float(prediction_proba[1] * 100)

    return {
        "prediction": prediction,
        "quality_label": quality_label,
        "confidence": round(confidence, 2),
    }
