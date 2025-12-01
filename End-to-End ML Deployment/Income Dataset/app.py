from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("knn_tuned_model.pkl", "rb") as f:
    knn_tuned = pickle.load(f)
    model = knn_tuned["model"]
    scaler = knn_tuned["scaler"]
    encoder = knn_tuned["encoder"]


class IncomeInput(BaseModel):
    age: int
    workclass: str
    finalweight: int  # Changed from fnlwgt
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    # Removed income field since it's not in the form


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(input_data: IncomeInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # Preprocess the input data using the trained encoder
    input_encoded = encoder.transform(input_df)
    input_scaled = scaler.transform(input_encoded)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_label = ">50K" if prediction[0] == 1 else "<=50K"

    return {"prediction": prediction_label}
