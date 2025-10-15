"""Train a RandomForestRegressor for the auto_mpg dataset and save artifacts.

Saves to repository root:
- mlp.pickle
- features_scaler.pickle

Run as a module to allow relative imports:
python -m homework.train_rf
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "auto_mpg.csv"
OUT_MLP = ROOT / "mlp.pickle"
OUT_SCALER = ROOT / "features_scaler.pickle"


def load_and_preprocess(path: Path):
    df = pd.read_csv(path)
    df = df.dropna()
    df["Origin"] = df["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    df = pd.get_dummies(df, columns=["Origin"], prefix="", prefix_sep="")
    y = df.pop("MPG")
    X = df
    return X, y


def train(random_state: int = 42):
    X, y = load_and_preprocess(INPUT)

    scaler = StandardScaler()
    # Fit scaler on the DataFrame so sklearn records feature names and
    # no warning is emitted later when transform is called with a DataFrame.
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(X_scaled, y.values)

    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y.values, y_pred)
    print(f"Train-set MSE (RF): {mse:.6f}")

    with open(OUT_MLP, "wb") as f:
        pickle.dump(model, f)

    with open(OUT_SCALER, "wb") as f:
        pickle.dump(scaler, f)

    return mse


if __name__ == "__main__":
    train()
