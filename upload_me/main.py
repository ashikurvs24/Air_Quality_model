import joblib
import pandas as pd
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Air Quality Model API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = joblib.load("model.pkl")

COLUMNS = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
           'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH',
           'day_of_month', 'month', 'year']

class Features(BaseModel):
    values: Dict[str, float]

@app.post("/predict")
def predict(data: Features):
    missing = [c for c in COLUMNS if c not in data.values]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing: {missing}")
    X = pd.DataFrame([[data.values[c] for c in COLUMNS]], columns=COLUMNS)
    pred = model.predict(X)[0]
    return {"prediction": float(pred) if not isinstance(pred, str) else pred}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")