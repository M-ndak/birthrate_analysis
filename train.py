import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

CSV_FILE = "japan_birth_statistics_1899_2023.csv"

df = pd.read_csv(CSV_FILE)
print("Columns:", df.columns.tolist())
print(df.head())

df = df[["year", "total_births"]].dropna()
df.columns = ["year", "births"]

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["births"] = pd.to_numeric(df["births"], errors="coerce")
df = df.dropna()

X = df[["year"]].values
y = df["births"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/birth_model.joblib")
print("Model saved to model/birth_model.joblib")
