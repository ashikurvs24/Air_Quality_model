# Air Quality Prediction App

A FastAPI app that serves a trained scikit-learn model with a mobile-friendly web interface.

## What it predicts
(write your target column here)

## Input features
CO(GT), PT08.S1(CO), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT),
PT08.S4(NO2), PT08.S5(O3), T, RH, plus a date (day, month, year).

## Run locally
pip install -r requirements.txt
uvicorn main:app --reload
Then open http://127.0.0.1:8000

## Live demo

