from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import subprocess
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
import numpy as np
import json

from src.app.schema.features_schema import WineFeatures

PROJECT_ROOT = Path.cwd()
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")
USE_DVC = os.getenv("USE_DVC")


def load_model_and_metrics(app: FastAPI):   
    model = None
    metrics = None
    
    if USE_DVC and int(USE_DVC) == 1:
        try:
            subprocess.run(["dvc", "pull", '--force', "models/model.pkl.dvc"], check=True, cwd=PROJECT_ROOT)
            logging.info("DVC pull successful for model")
        except subprocess.CalledProcessError as e:
            logging.warning("DVC pull failed for model: %s", e)
        except FileNotFoundError:
            logging.warning("DVC not installed or model .dvc file not found")
        
        try:
            subprocess.run(["dvc", "pull", "models/metrics.json.dvc"], check=True, cwd=PROJECT_ROOT)
            logging.info("DVC pull successful for metrics")
        except subprocess.CalledProcessError as e:
            logging.warning("DVC pull failed for metrics: %s", e)
        except FileNotFoundError:
            logging.warning("DVC not installed or metrics .dvc file not found")
    
    try:
        model = joblib.load(MODEL_PATH)  # Используем MODEL_PATH
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.exception("Failed to load model: %s", e)
        model = None
    
    try:
        with open(METRICS_PATH, 'r') as f:  # Используем METRICS_PATH
            metrics = json.load(f)
        logging.info("Metrics loaded successfully")
    except Exception as e:
        logging.exception("Failed to load metrics: %s", e)
        metrics = None
    
    return model, metrics



@asynccontextmanager
async def lifespan(app: FastAPI):
    model, metrics = load_model_and_metrics(app)
    app.state.model = model
    app.state.metrics = metrics
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(request: Request, data: WineFeatures) -> dict:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    feature_mapping = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol"
    }
    
    expected_features = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol"
    ]
    
    data_dict = data.model_dump()
    mapped_data = {feature_mapping[key]: value for key, value in data_dict.items()}
    features = pd.DataFrame([mapped_data], columns=expected_features)
    
    if not all(key in mapped_data for key in expected_features):
        raise HTTPException(status_code=400, detail="Missing or invalid features after mapping")
    
    try:
        prediction = model.predict(features)[0]
        result = {"prediction": float(prediction) if isinstance(prediction, (int, float)) else int(prediction)}
        return result
    except Exception as e:
        logging.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
def health(request: Request) -> dict:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status}

@app.get("/model-info")
def model_info(request: Request) -> dict:
    model = request.app.state.model
    metrics = request.app.state.metrics
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        info = {
            "model_type": type(model).__name__,
            "model_path": MODEL_PATH,
            "is_trained": hasattr(model, "predict"),
            "n_estimators": getattr(model, 'n_estimators', None),
            "max_depth": getattr(model, 'max_depth', None),
            "min_samples_split": getattr(model, 'min_samples_split', None),
            "min_samples_leaf": getattr(model, 'min_samples_leaf', None),
            "random_state": getattr(model, 'random_state', None),
            "criterion": getattr(model, 'criterion', None),
            "n_features_in": getattr(model, 'n_features_in_', None),
            "classes": getattr(model, 'classes_', None).tolist() if hasattr(model, 'classes_') else None,
            "metrics": metrics
        }
        return info
    except Exception as e:
        logging.exception("Model info error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get model info")

@app.get("/")
def root():
    return {"message": "Wine Classification API"}

if __name__ == '__main__':
    uvicorn.run(app='api:app', host="0.0.0.0", port=8000)
