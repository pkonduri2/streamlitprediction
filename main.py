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
dmr_col = "DMR_ID"             # DMR identifier
feature_col = "pfas_exposure"  # PFAS column
target_col = "meth.diff"       # Methylation difference column

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
# 4. Predictive model (PFAS → methylation)
# -----------------------------
st.subheader(f"Predictive model for DMR: {dmr_selected}")
X = df_dmr[[feature_col]]
y = df_dmr[target_col]

model = LinearRegression()
model.fit(X, y)

pfas_value = st.number_input(
    "Enter PFAS exposure",
    value=float(X[feature_col].median())
)

if st.button("Predict methylation for this DMR"):
    input_df = pd.DataFrame({feature_col: [pfas_value]})
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted methylation difference: {prediction:.4f}")

with st.expander("Predictive model details"):
    st.write("Intercept:", model.intercept_)
    st.write("Slope (PFAS effect):", model.coef_[0])
    st.write("Training samples:", len(df_dmr))

# -----------------------------
# 5. Causal / perturbation approximation
# -----------------------------
st.subheader(f"Causal / perturbation approximation for {dmr_selected}")
st.write("""
We approximate the causal effect of PFAS on methylation using the slope of the predictive model.
This allows simulating how perturbing methylation might affect downstream outcomes.
""")

# Approximate causal effect using linear regression slope
causal_effect = model.coef_[0]
st.write("Estimated causal effect of PFAS on methylation (approx.):", causal_effect)

# Simulate methylation perturbation
st.subheader("Simulate methylation perturbation")
perturb = st.slider("Change methylation (delta)", -0.5, 0.5, 0.0)
predicted_outcome = perturb * causal_effect
st.write(f"Predicted change in outcome if methylation is perturbed: {predicted_outcome:.4f}")
