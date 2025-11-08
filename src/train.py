import os
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")

def train_model(df: pd.DataFrame):
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config["model"]
    X = df.drop(columns=["quality"])
    y = df["quality"]
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
    )
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return MODEL_PATH

if __name__ == "__main__":
    df = pd.read_csv("data/winequality-red.csv")
    train_model(df)
