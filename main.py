import warnings
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Suppress minor version warnings from scikit-learn model loading
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(
    title="Iris Flower Species Classifier",
    description="FastAPI Web Application using pre-trained machine learning model artifacts",
    version="1.0.0"
)

# Base directory for loading model files
BASE_DIR = Path(__file__).resolve().parent

# Load ML artifacts
MODEL_PATH = BASE_DIR / "model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"
ENCODER_PATH = BASE_DIR / "encoder.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("[OK] Model, Scaler, and Encoder successfully loaded!")
except Exception as e:
    print(f"[ERROR] Error loading model artifacts: {e}")
    model, scaler, encoder = None, None, None

# Templates setup
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Pydantic input schema
class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Sepal Length in cm")
    sepal_width: float = Field(..., example=3.5, description="Sepal Width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal Length in cm")
    petal_width: float = Field(..., example=0.2, description="Petal Width in cm")

def perform_prediction(sepal_len: float, sepal_w: float, petal_len: float, petal_w: float) -> Dict[str, Any]:
    if model is None or scaler is None or encoder is None:
        raise ValueError("Model artifacts are not loaded properly.")

    # Feature names expected by fitted scaler & model
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    input_df = pd.DataFrame([[sepal_len, sepal_w, petal_len, petal_w]], columns=feature_names)
    
    # Scale input features
    scaled_features = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
    
    # Predict species and class probabilities
    pred_class_idx = model.predict(scaled_df)
    probabilities = model.predict_proba(scaled_df)[0]
    
    # Decode species class label
    species_label = encoder.inverse_transform(pred_class_idx)[0]
    
    # Map probability per class label
    classes = encoder.classes_
    prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)}
    max_confidence = round(float(np.max(probabilities)), 4)
    
    return {
        "prediction": str(species_label),
        "confidence": max_confidence,
        "probabilities": prob_dict,
        "input": {
            "sepal_length": sepal_len,
            "sepal_width": sepal_w,
            "petal_length": petal_len,
            "petal_width": petal_w
        }
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main interactive web page."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None}
    )

@app.post("/predict")
async def predict_api(data: IrisInput):
    """REST API Endpoint for Iris Prediction (JSON input & output)."""
    try:
        result = perform_prediction(
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        )
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    """Form submission endpoint for standard HTML form posting."""
    try:
        result = perform_prediction(sepal_length, sepal_width, petal_length, petal_width)
    except Exception as e:
        result = {"error": str(e)}
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "form_data": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
