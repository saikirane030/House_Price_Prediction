from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "house_model.pkl"
FEATURES = [
    "Gr Liv Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Year Built",
    "Kitchen Qual",
    "Garage Cars",
]
NUMERIC_FEATURES = [
    "Gr Liv Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Year Built",
    "Garage Cars",
]
KITCHEN_QUALITIES = {
    "Po": "Poor",
    "Fa": "Fair",
    "TA": "Typical / Average",
    "Gd": "Good",
    "Ex": "Excellent",
}

st.set_page_config(
    page_title="Ames House Estimate",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ivory: #f5f1e8;
        --paper: #fffdf8;
        --evergreen: #173d32;
        --evergreen-light: #245343;
        --brass: #b58a45;
        --ink: #20312b;
        --muted: #69736c;
        --line: #e5dfd2;
    }
    .stApp { background: var(--ivory); color: var(--ink); }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    .hero {
        background: linear-gradient(120deg, var(--evergreen), var(--evergreen-light));
        color: #fffdf8;
        border-radius: 24px;
        padding: clamp(1.6rem, 4vw, 3.2rem);
        margin-bottom: 1.6rem;
        box-shadow: 0 16px 38px rgba(23, 61, 50, .14);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -45px;
        top: -95px;
        border: 1px solid rgba(220, 190, 133, .34);
        border-radius: 50%;
        box-shadow: 0 0 0 25px rgba(220, 190, 133, .06),
                    0 0 0 55px rgba(220, 190, 133, .04);
    }
    .eyebrow {
        color: #e0c28a;
        font-size: .77rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .hero h1 {
        color: #fffdf8;
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 500;
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        letter-spacing: -.035em;
        line-height: 1.06;
        margin: 0 0 .8rem 0;
    }
    .hero p { color: rgba(255, 253, 248, .82); max-width: 690px; margin: 0; }
    .section-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.55rem;
        color: var(--evergreen);
        margin: .4rem 0 .2rem;
    }
    .section-copy { color: var(--muted); margin-bottom: 1rem; }
    div[data-testid="stForm"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.15rem 1.25rem .75rem;
    }
    div[data-testid="stForm"] label p { color: var(--ink); font-weight: 650; }
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] p { margin-bottom: .15rem; }
    div[data-testid="stForm"] [data-testid="stSlider"] { padding-top: .4rem; }
    div[data-testid="stForm"] input, div[data-testid="stForm"] [data-baseweb="select"] > div {
        background: #fffefa;
        border-color: var(--line);
        border-radius: 10px;
    }
    .result-card {
        background: var(--evergreen);
        color: #fffdf8;
        padding: clamp(1.5rem, 3vw, 2.3rem);
        border-radius: 20px;
        margin-top: 1.3rem;
        box-shadow: 0 14px 32px rgba(23, 61, 50, .13);
    }
    .result-label {
        color: #e0c28a;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .result-price {
        color: #fffdf8;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        line-height: 1.15;
        margin: .25rem 0 .75rem;
    }
    .range-line { color: rgba(255, 253, 248, .87); font-size: 1.05rem; }
    .result-note { color: rgba(255, 253, 248, .7); font-size: .88rem; margin-top: .85rem; }
    .metric-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-top: 3px solid var(--brass);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        min-height: 94px;
    }
    .metric-name {
        color: var(--muted);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .metric-value { color: var(--evergreen); font-size: 1.35rem; font-weight: 700; margin-top: .25rem; }
    .footer-note {
        color: var(--muted);
        font-size: .84rem;
        border-top: 1px solid var(--line);
        margin-top: 2rem;
        padding-top: 1rem;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: var(--brass);
        color: #fffdf8;
        border: 0;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
        padding: .55rem 1.2rem;
        transition: background .15s ease, transform .15s ease;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #9b7335;
        color: white;
        transform: translateY(-1px);
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .hero { border-radius: 18px; }
        div[data-testid="stForm"] { padding: .8rem .8rem .5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_package():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Train/export house_model.pkl first."
        )
    package = joblib.load(MODEL_PATH)
    if not isinstance(package, dict):
        raise TypeError("house_model.pkl must contain a dictionary model package.")
    if package.get("features") != FEATURES:
        raise ValueError(
            "The saved model feature list is incompatible. "
            f"Expected {FEATURES}, got {package.get('features')}."
        )
    if not hasattr(package.get("model"), "predict"):
        raise TypeError("The saved package does not contain a usable prediction model.")
    if not all(key in package for key in ("medians", "lower_bound", "upper_bound", "metrics")):
        raise ValueError(
            "The model package must include medians, lower_bound, upper_bound, and metrics."
        )
    return package


try:
    model_package = load_package()
except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
    st.error(f"Estimator unavailable: {exc}")
    st.info(
        "This estimator requires a six-feature pipeline trained on the Ames dataset. "
        "It will not make predictions from an incompatible model."
    )
    st.stop()

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">A considered estimate · Ames, Iowa</div>
      <h1>Find your home's<br>starting point.</h1>
      <p>A thoughtful price estimate informed by historical Ames home sales.
      Share a few details to see an estimated value and a range that reflects model uncertainty.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

form_col, spacer_col, detail_col = st.columns([1.05, 0.08, 0.95], gap="large")

with form_col:
    st.markdown('<div class="section-title">Tell us about the home</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A few simple details help shape your estimate.</div>',
        unsafe_allow_html=True,
    )

    medians = model_package["medians"]
    with st.form("home_details"):
        gr_liv_area = st.number_input(
            "Living area (sq ft)",
            min_value=334,
            max_value=5642,
            value=int(round(medians["Gr Liv Area"])),
            step=25,
            help="The total above-ground living space in the home, in square feet.",
        )
        overall_qual = st.slider(
            "Overall quality",
            min_value=1,
            max_value=10,
            value=int(round(medians["Overall Qual"])),
            help="Ames home's overall material and finish rating: 1 is very poor; 10 is excellent.",
        )
        total_bsmt_sf = st.number_input(
            "Basement area (sq ft)",
            min_value=0,
            max_value=6110,
            value=int(round(medians["Total Bsmt SF"])),
            step=25,
            help="Total basement area in square feet. Enter 0 if the home has no basement.",
        )
        year_built = st.number_input(
            "Year built",
            min_value=1872,
            max_value=2010,
            value=int(round(medians["Year Built"])),
            step=1,
            help="The original year the home was constructed, within the dataset's historical range.",
        )
        kitchen_quality_label = st.selectbox(
            "Kitchen quality",
            options=list(KITCHEN_QUALITIES.values()),
            index=list(KITCHEN_QUALITIES.values()).index("Typical / Average"),
            help="Choose the kitchen's overall material and finish quality.",
        )
        kitchen_qual = next(
            code for code, label in KITCHEN_QUALITIES.items()
            if label == kitchen_quality_label
        )
        garage_cars = st.number_input(
            "Garage capacity (cars)",
            min_value=0,
            max_value=5,
            value=int(round(medians["Garage Cars"])),
            step=1,
            help="How many cars the garage can hold. Enter 0 if there is no garage.",
        )
        submitted = st.form_submit_button("Estimate my home's value", use_container_width=True)

with detail_col:
    st.markdown('<div class="section-title">A clear, considered range</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Your estimate is based on historical Ames, Iowa sales and shown in USD.</div>',
        unsafe_allow_html=True,
    )
    if submitted:
        input_data = pd.DataFrame(
            [[
                gr_liv_area,
                overall_qual,
                total_bsmt_sf,
                year_built,
                kitchen_qual,
                garage_cars,
            ]],
            columns=FEATURES,
        )
        estimate = float(model_package["model"].predict(input_data)[0])
        low = max(0.0, estimate + float(model_package["lower_bound"]))
        high = max(low, estimate + float(model_package["upper_bound"]))
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Estimated best value</div>
              <div class="result-price">${estimate:,.0f}</div>
              <div class="range-line">Estimated range: <strong>${low:,.0f} – ${high:,.0f}</strong></div>
              <div class="result-note">The range is calibrated from held-out Ames sales residuals.
              It communicates uncertainty and is not a guarantee or formal appraisal.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("See the exact input sent to the model"):
            st.dataframe(input_data, hide_index=True, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="result-card">
              <div class="result-label">Your estimate will appear here</div>
              <div class="result-price">Ready when you are</div>
              <div class="range-line">Complete the home details and select the brass button to continue.</div>
              <div class="result-note">This tool estimates historical Ames, Iowa sale prices in USD.
              It is not an appraisal or a valuation for another market.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title" style="margin-top:2rem">How the model performs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Fresh held-out test results from the saved six-factor model.</div>',
    unsafe_allow_html=True,
)

metrics = model_package["metrics"]
metric_columns = st.columns(4, gap="medium")
metric_values = [
    ("Mean absolute error", f"${metrics['test_mae']:,.0f}"),
    ("Root mean squared error", f"${metrics['test_rmse']:,.0f}"),
    ("R² score", f"{metrics['test_r2']:.3f}"),
    (
        "Typical estimate uncertainty",
        f"−${abs(float(model_package['lower_bound'])):,.0f} / +${float(model_package['upper_bound']):,.0f}",
    ),
]
for column, (label, value) in zip(metric_columns, metric_values):
    column.markdown(
        f'<div class="metric-card"><div class="metric-name">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

coverage = metrics.get("test_interval_coverage")
interval_level = metrics.get("interval_level", 0.90)
coverage_note = (
    f" On the held-out test set, the calibrated range covered "
    f"{coverage:.0%} of actual sale prices (nominal {interval_level:.0%})."
    if coverage is not None
    else ""
)
st.markdown(
    f"""
    <div class="footer-note">
      Estimates reflect historical Ames, Iowa sales and are denominated in USD.
      MAE and RMSE summarize test-set prediction error; R² describes test-set variance explained.
      The displayed range is an uncertainty aid, not a guarantee, appraisal, or Indian-market valuation.
      {coverage_note}
    </div>
    """,
    unsafe_allow_html=True,
)
