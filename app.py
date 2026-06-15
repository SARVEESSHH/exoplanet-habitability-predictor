import streamlit as st
import pandas as pd
import joblib

# =====================
# LOAD MODEL
# =====================

model = joblib.load(
    "models/habitability_model.pkl"
)

# =====================
# PAGE CONFIG
# =====================

st.set_page_config(
    page_title="AI Exoplanet Habitability Ranking",
    page_icon="🌍",
    layout="wide"
)

# =====================
# TITLE
# =====================

st.title(
    "🌍 AI + Earth Similarity Exoplanet Habitability Ranking System"
)

st.write(
    """
    Predict the habitability potential of an exoplanet
    using a Random Forest machine learning model trained
    on real NASA exoplanet data.
    """
)

# =====================
# INPUTS
# =====================

st.header("🪐 Enter Planet Parameters")

col1, col2 = st.columns(2)

with col1:

    radius = st.number_input(
        "Planet Radius (Earth Radii)",
        value=1.0
    )

    mass = st.number_input(
        "Planet Mass (Earth Masses)",
        value=1.0
    )

    period = st.number_input(
        "Orbital Period (Days)",
        value=365.0
    )

with col2:

    temperature = st.number_input(
        "Equilibrium Temperature (K)",
        value=288.0
    )

    star_temp = st.number_input(
        "Star Temperature (K)",
        value=5800.0
    )

    star_radius = st.number_input(
        "Star Radius (Solar Radii)",
        value=1.0
    )

# =====================
# PREDICTION
# =====================

if st.button("🚀 Predict Habitability"):

    sample = pd.DataFrame({

        "radius":[radius],
        "mass":[mass],
        "period":[period],
        "temperature":[temperature],
        "star_temp":[star_temp],
        "star_radius":[star_radius]

    })

    score = model.predict_proba(sample)[0][1]

    st.success(
        f"Habitability Score: {score:.2%}"
    )

# =====================
# PROJECT IMAGES
# =====================

st.header("📊 Project Results")

st.image(
    "images/top_habitable_planets.png"
)

st.image(
    "images/feature_importance.png"
)

st.image(
    "images/planet_radius_distribution.png"
)
