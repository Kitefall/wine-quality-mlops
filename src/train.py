import os
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")

def train_model(df: pd.DataFrame):
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config["model"]


    X = df.drop(columns=["quality"])
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params["random_state"])
    
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")

    return MODEL_PATH

if __name__ == "__main__":
    df = pd.read_csv("data/winequality-red.csv")
    train_model(df)
