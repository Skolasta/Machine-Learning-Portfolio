import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

app = FastAPI()

# Şablonlar için klasör
templates = Jinja2Templates(directory="templates")

# Modeli yükle
MODEL_PATH = "rf_telco_churn_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Model dosyası yoksa hata fırlatmayalım, sadece uyaralım (geliştirme aşaması için)
    print(f"UYARI: {MODEL_PATH} bulunamadı. Tahmin fonksiyonu çalışmayacaktır.")
    model = None

# Pydantic modeli (API testi için)
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def feature_engineering(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    # Sayısal dönüşümler
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)

    # Feature engineering - Basic
    df['IsMonthToMonth'] = np.where(df['Contract'] == 'Month-to-month', 1, 0)
    df['IsAutoPay'] = np.where(df['PaymentMethod'].str.contains('automatic'), 1, 0)
    df['HasPhoneService'] = np.where(df['PhoneService'] == 'Yes', 1, 0)
    df['HasInternetService'] = np.where(df['InternetService'] == 'No', 0, 1)
    df['IsAlone'] = np.where((df['Partner'] == 'No') & (df['Dependents'] == 'No'), 1, 0)

    # Feature engineering - Financial
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['PriceIncrease'] = df['MonthlyCharges'] / (df['AvgMonthlyCharges'] + 0.01)
    df['ExpectedRevenue'] = df['MonthlyCharges'] * 12

    # Feature engineering - Service Combinations
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # "Yes" sayısını hesapla
    df['TotalServices'] = 0
    for col in service_cols:
        df['TotalServices'] += (df[col] == 'Yes').astype(int)
        
    df['IsPremiumCustomer'] = (df['TotalServices'] >= 4).astype(int)
    df['NoExtraServices'] = (df['TotalServices'] == 0).astype(int)

    # Feature engineering - Risk Factors
    df['HighRiskProfile'] = ((df['Contract'] == 'Month-to-month') & 
                              (df['PaymentMethod'] == 'Electronic check')).astype(int)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    df['LoyaltyScore'] = df['tenure'] * (3 - df['IsMonthToMonth'] * 2)
    df['FirstYear'] = (df['tenure'] <= 12).astype(int)

    # Feature engineering - Internet Patterns
    df['FiberNoSecurity'] = ((df['InternetService'] == 'Fiber optic') & 
                              (df['OnlineSecurity'] == 'No')).astype(int)

    # Feature engineering - Demographics
    df['SeniorAlone'] = ((df['SeniorCitizen'] == 1) & (df['IsAlone'] == 1)).astype(int)
    df['YoungFamily'] = ((df['SeniorCitizen'] == 0) & 
                          (df['Partner'] == 'Yes') & 
                          (df['Dependents'] == 'Yes')).astype(int)
    
    return df

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)
):
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        if model is None:
            raise ValueError("Model dosyası yüklenemediği için tahmin yapılamıyor.")

        df = feature_engineering(input_data)
        
        # Tahmin
        prediction = model.predict(df)[0]
        # predict_proba kontrolü
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df)[0][1]
        else:
            # Pipeline içindeki son adımın predict_proba'sı var mı?
            try:
                probability = model.predict_proba(df)[0][1]
            except:
                probability = 0.0 # Desteklenmiyorsa
        
        result = "Churn (Terk Edecek)" if prediction == 1 else "No Churn (Kalacak)"
        color = "red" if prediction == 1 else "green"
        prob_percent = round(probability * 100, 2)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result,
            "probability": prob_percent,
            "color": color,
            "input_data": input_data
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "input_data": input_data
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Model updated trigger
