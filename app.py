from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "birth_model.joblib")
model = joblib.load(MODEL_PATH)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Japan Births Predictor</title>
<style>
  :root {
    --red: #d62828;
    --blue: #1d63d1;
    --bg: #f7f5f0;
    --card: #ffffff;
    --text: #1a1a1a;
    --muted: #5a5a5a;
    --border: #e5e2da;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }
  .wrap {
    max-width: 820px;
    margin: 0 auto;
    padding: 40px 20px 70px;
  }
  header h1 {
    font-size: 2rem;
    margin: 0 0 6px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }
  header p.subtitle {
    margin: 0 0 30px;
    color: var(--muted);
    font-size: 1.05rem;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  }
  .predict-form {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    flex-wrap: wrap;
  }
  .field label {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: var(--muted);
  }
  input[type="number"] {
    padding: 12px 14px;
    font-size: 1rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    width: 160px;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type="number"]:focus { border-color: var(--blue); }
  button {
    padding: 12px 22px;
    font-size: 1rem;
    font-weight: 600;
    background: var(--blue);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #164fae; }
  .result {
    margin-top: 22px;
    padding: 18px 20px;
    background: #eef4ff;
    border-left: 4px solid var(--blue);
    border-radius: 8px;
    font-size: 1.1rem;
  }
  .result strong { color: var(--blue); font-size: 1.3rem; }
  .stats-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 18px;
  }
  .stat {
    flex: 1;
    min-width: 140px;
    background: #faf8f4;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
  }
  .stat .num {
    font-size: 1.4rem;
    font-weight: 800;
  }
  .stat.red .num { color: var(--red); }
  .stat.blue .num { color: var(--blue); }
  .stat .label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 4px;
  }
  h2 {
    font-size: 1.3rem;
    margin: 0 0 14px;
  }
  .chart-img {
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--border);
    display: block;
    margin-top: 14px;
  }
  .context p { margin: 0 0 14px; }
  .context ul { margin: 0 0 14px 20px; padding: 0; }
  .context li { margin-bottom: 8px; }
  .source {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 10px;
  }
  footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 30px;
  }
  code.inline {
    background: #f0eee8;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Japan Births Predictor</h1>
    <p class="subtitle">A Random Forest model trained on Japan's official birth statistics (1899&ndash;2023) predicts total annual births for a given year.</p>
  </header>

  <div class="card">
    <h2>Try a prediction</h2>
    <form class="predict-form" action="/predict" method="get">
      <div class="field">
        <label for="year">Year</label>
        <input type="number" id="year" name="year" value="{{ year if year else 2025 }}" required>
      </div>
      <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
    <div class="result">
      Predicted total births in <strong>{{ year }}</strong>: <strong>{{ "{:,}".format(prediction) }}</strong>
    </div>
    {% endif %}

    <div class="stats-row">
      <div class="stat red">
        <div class="num">1.62M</div>
        <div class="label">Deaths in 2024</div>
      </div>
      <div class="stat blue">
        <div class="num">721K</div>
        <div class="label">Births in 2024 (record low)</div>
      </div>
      <div class="stat">
        <div class="num">2005</div>
        <div class="label">Year deaths first exceeded births</div>
      </div>
    </div>
  </div>

  <div class="card context">
    <h2>Background: Japan's declining birthrate</h2>
    <p>
      Japan recorded its lowest number of annual births on record in 2024, continuing a
      decades-long decline that began after the post-war baby boom of the late 1940s.
      Deaths, meanwhile, have been climbing steadily as the population ages &mdash; the two
      trends crossed in 2005, when deaths outpaced births for the first time in the
      modern era, and the gap has widened every year since.
    </p>
    <ul>
      <li><strong>1947&ndash;1949:</strong> Post-war baby boom peaks at over 2.6 million births a year.</li>
      <li><strong>1966:</strong> Births fall by roughly 25% in a single year &mdash; the year of the <em>hinoe-uma</em> (Fire Horse), a superstition linked to bad luck for daughters born that year.</li>
      <li><strong>2005:</strong> Deaths exceed births for the first time, marking the start of sustained natural population decline.</li>
      <li><strong>2024:</strong> Births fall to roughly 721,000 &mdash; a new record low &mdash; while deaths reach about 1.62 million.</li>
    </ul>
    <p>
      This model captures the non-linear shape of that curve &mdash; rise, plateau, and
      decline &mdash; using a Random Forest regressor, which fits the data far better than a
      simple straight-line trend (R&sup2; &asymp; 0.96 vs. &asymp; 0.78&ndash;0.84 for polynomial
      regression on the same data).
    </p>
    <img class="chart-img" src="/static/japan_births_chart.png" alt="Chart showing Japan's annual births and deaths from 1947 to 2024">
    <div class="source">Source: Ministry of Health, Labour and Welfare of Japan &mdash; chart via Chartr</div>
  </div>

  <div class="card">
    <h2>API usage</h2>
    <p>Prefer JSON? Call the same endpoint directly:</p>
    <p><code class="inline">GET /predict?year=2025</code></p>
    <p>Returns: <code class="inline">{"year": 2025, "predicted_births": 700000}</code></p>
  </div>

  <footer>Model: Random Forest Regressor &middot; Trained on 1899&ndash;2023 Japan birth data</footer>
</div>
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