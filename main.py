import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="PFAS–Methylation Predictor + Causal Approximation")

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("pfas_dmr_data.csv")

st.title("PFAS Exposure → DNA Methylation Predictor & Causal Approximation")
st.write("Dataset preview:")
st.dataframe(df.head())

# -----------------------------
# 2. Define columns
# -----------------------------
dmr_col = "DMR_ID"
feature_col = "pfas_exposure"
target_col = "meth.diff"

for col in [dmr_col, feature_col, target_col]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in dataset.")
        st.stop()

# -----------------------------
# 3. Filter DMRs with enough samples
# -----------------------------
dmr_counts = df.groupby(dmr_col).size()
valid_dmrs = dmr_counts[dmr_counts >= 2].index.tolist()

if not valid_dmrs:
    st.error("No DMRs have enough data (≥2 samples) to train a model.")
    st.stop()

st.subheader("Select a DMR")
dmr_selected = st.selectbox("Choose DMR", valid_dmrs)
df_dmr = df[df[dmr_col] == dmr_selected][[feature_col, target_col]].dropna()

# -----------------------------
# 4. Predictive model section
# -----------------------------
st.header("Predictive Model")
st.write("""
This section predicts methylation differences for the selected DMR based purely on PFAS exposure.
It does **not** make causal assumptions.
""")

X = df_dmr[[feature_col]]
y = df_dmr[target_col]

predictive_model = LinearRegression()
predictive_model.fit(X, y)

pfas_value = st.number_input(
    "Enter PFAS exposure for prediction",
    value=float(X[feature_col].median())
)

if st.button("Predict methylation"):
    input_df = pd.DataFrame({feature_col: [pfas_value]})
    prediction = predictive_model.predict(input_df)[0]
    st.success(f"Predicted methylation difference: {prediction:.4f}")

with st.expander("Predictive Model Details"):
    st.write("Intercept:", predictive_model.intercept_)
    st.write("Slope (PFAS effect):", predictive_model.coef_[0])
    st.write("Training samples:", len(df_dmr))

# -----------------------------
# 5. Causal / perturbation approximation section
# -----------------------------
st.header("Causal / Perturbation Approximation")
st.write("""
This section **approximates a causal effect** of PFAS on methylation using the slope of the predictive model.
It allows simulating how perturbing methylation might affect downstream outcomes.
""")

# Use slope of predictive model as causal effect
causal_effect = predictive_model.coef_[0]
st.write("Approximate causal effect of PFAS on methylation:", causal_effect)

# Simulate methylation perturbation
st.subheader("Simulate methylation perturbation")
perturb = st.slider("Change methylation (delta)", -0.5, 0.5, 0.0)
predicted_outcome = perturb * causal_effect
st.write(f"Predicted change in outcome if methylation is perturbed: {predicted_outcome:.4f}")
