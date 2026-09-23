<<<<<<< HEAD
=======
import kagglehub
>>>>>>> dafb54910514679376665084a32a07afd0de0978
import pandas as pd
import numpy as np
import os
import joblib
<<<<<<< HEAD
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
=======
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Download dataset
path = kagglehub.dataset_download("hideos/japan-birth-statistics-18992023")
print("Path to dataset files:", path)

# Find the CSV file inside the downloaded folder
csv_file = None
for f in os.listdir(path):
    if f.endswith(".csv"):
        csv_file = os.path.join(path, f)
        break

if csv_file is None:
    raise FileNotFoundError("No CSV found in dataset folder. Check contents: " + str(os.listdir(path)))

print("Using CSV:", csv_file)

df = pd.read_csv(csv_file)
print("Columns:", df.columns.tolist())
print(df.head())

# NOTE: Adjust these column names to match your actual CSV after checking the printout above.
year_col = [c for c in df.columns if "year" in c.lower()][0]
birth_col = [c for c in df.columns if "birth" in c.lower()][0]

df = df[[year_col, birth_col]].dropna()
df.columns = ["year", "births"]

# Clean non-numeric junk (commas, strings) if present
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["births"] = pd.to_numeric(df["births"].astype(str).str.replace(",", ""), errors="coerce")
>>>>>>> dafb54910514679376665084a32a07afd0de0978
df = df.dropna()

X = df[["year"]].values
y = df["births"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

<<<<<<< HEAD
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
)
=======
model = LinearRegression()
>>>>>>> dafb54910514679376665084a32a07afd0de0978
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

<<<<<<< HEAD
=======
# Save model
>>>>>>> dafb54910514679376665084a32a07afd0de0978
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/birth_model.joblib")
print("Model saved to model/birth_model.joblib")
