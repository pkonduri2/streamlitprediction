import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="PFAS → Methylation Model")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("pfas_dmr_data.csv")

st.title("PFAS Exposure → DNA Methylation Model")

st.write("Dataset preview:")
st.dataframe(df.head())

# -----------------------------
# Define columns
# -----------------------------
feature_col = "pfas_exposure"
target_col = "meth.diff"

if feature_col not in df.columns or target_col not in df.columns:
    st.error("Required columns not found in dataset.")
    st.stop()

# -----------------------------
# Clean data
# -----------------------------
df_model = df[[feature_col, target_col]].dropna()

if len(df_model) < 2:
    st.error("Not enough data to train model.")
    st.stop()

X = df_model[[feature_col]]
y = df_model[target_col]

# -----------------------------
# Predictive model
# -----------------------------
st.header("Predictive Model")

model = LinearRegression()
model.fit(X, y)

pfas_value = st.number_input(
    "Enter PFAS exposure",
    value=float(X[feature_col].median())
)

if st.button("Predict methylation difference"):
    input_df = pd.DataFrame({feature_col: [pfas_value]})
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted methylation difference: {prediction:.4f}")

with st.expander("Predictive model details"):
    st.write("Model type: Linear regression")
    st.write("Intercept:", model.intercept_)
    st.write("Slope (PFAS effect):", model.coef_[0])
    st.write("Number of DMRs used:", len(df_model))

# -----------------------------
# Causal / perturbation interpretation
# -----------------------------
st.header("Causal / Perturbation Interpretation")

st.write("""
This section interprets the regression coefficient as an **approximate causal effect**.
This does not prove causality, but estimates the expected methylation response to PFAS perturbation.
""")

causal_effect = model.coef_[0]
st.write("Approximate causal effect (Δmethylation per unit PFAS):", causal_effect)

delta_pfas = st.slider("Simulate PFAS increase", 0.0, 5.0, 1.0)
estimated_change = delta_pfas * causal_effect
st.write(f"Estimated methylation change: {estimated_change:.4f}")
