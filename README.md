# Japan Births Predictor

A Flask web app serving a Random Forest regression model trained on Japan
birth statistics (1899-2023) to predict total births for a given year.

## Project structure

```
.
├── app.py
├── train.py
├── japan_birth_statistics_1899_2023.csv
├── model/
│   └── birth_model.joblib
├── requirements.txt
├── render.yaml
├── LICENSE
└── README.md
```

## Model

Random Forest Regressor (300 trees) on year -> total_births.
On a held-out 20% test split: **R^2 ~ 0.956, MAE ~ 49,000 births**.
This captures the non-linear rise-then-decline pattern in Japan's births
much better than plain linear or low-degree polynomial regression (which
topped out around R^2 ~ 0.78-0.84 on the same split).

Caveat: Random Forest predictions for years far outside the training range
(e.g. 2050+) will just flatten out near the nearest training years' values,
since trees cannot extrapolate trends. It's most reliable for years within
or close to 1899-2023.

## License

MIT — see [LICENSE](LICENSE).