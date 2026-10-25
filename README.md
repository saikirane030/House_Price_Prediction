# Ames House Price Estimator

A Streamlit app that estimates historical Ames, Iowa home sale prices in USD.
It uses a saved scikit-learn pipeline, displays a calibrated prediction range,
and reports held-out test metrics. This is an educational estimate, not a
formal appraisal or a valuation for another market.

## Project files

- `app.py` — Streamlit interface and prediction flow.
- `house_model.pkl` — saved preprocessing-and-model package, input feature
  metadata, training medians, interval bounds, and test metrics.
- `housing_data.csv` — Ames housing data used by the project.
- `SaiKiran_LinearRegression_Project.ipynb` — notebook documenting the model
  development work.
- `requirements.txt` — Python dependencies.

## Run locally

Use Python 3.10 or later. From the project folder, install dependencies and
start the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Push this project to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from the repository's `main` branch and select `app.py`.
3. Deploy and share the public app URL.

The model file must remain in the repository beside `app.py`. Do not load pickle
files from untrusted sources.

## Inference flow

The form collects the following features in the order expected by the saved
model:

1. `Gr Liv Area`
2. `Overall Qual`
3. `Total Bsmt SF`
4. `Year Built`
5. `Kitchen Qual`
6. `Garage Cars`

The app creates a one-row pandas DataFrame, sends it to the saved pipeline's
`predict()` method, and displays the estimate with calibrated lower and upper
bounds. It also displays the package's held-out test MAE, RMSE, R², and interval
coverage.

## Notes

- The estimates are based on historical Ames, Iowa sales and are denominated
  in USD; they should not be interpreted as current market prices.
- The displayed range is an uncertainty aid, not a guarantee or appraisal.
- The app runs locally or as a hosted Streamlit app. It does not currently
  provide a separate REST API or ongoing production monitoring.
