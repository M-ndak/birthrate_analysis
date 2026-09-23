from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "birth_model.joblib")
model = joblib.load(MODEL_PATH)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Japan Births Predictor</title></head>
<body style="font-family: sans-serif; max-width: 500px; margin: 50px auto;">
  <h2>Japan Birth Count Predictor</h2>
  <form action="/predict" method="get">
    <label>Enter Year: </label>
    <input type="number" name="year" value="2025" required>
    <button type="submit">Predict</button>
  </form>
  {% if prediction is not none %}
    <h3>Predicted births in {{ year }}: {{ prediction }}</h3>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE, prediction=None, year=None)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET" and "year" in request.args and request.accept_mimetypes.accept_html:
        year = int(request.args.get("year"))
        pred = model.predict(np.array([[year]]))[0]
        return render_template_string(HTML_PAGE, prediction=round(float(pred)), year=year)

    data = request.get_json(silent=True) or request.args
    year = data.get("year")
    if year is None:
        return jsonify({"error": "Provide 'year' as a query param or JSON body"}), 400

    year = int(year)
    pred = model.predict(np.array([[year]]))[0]
    return jsonify({"year": year, "predicted_births": round(float(pred))})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
