import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import dowhy
from dowhy import CausalModel

st.set_page_config(page_title="PFAS–Methylation Predictor + Causal Model")

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("pfas_dmr_data.csv")

st.title("PFAS Exposure → DNA Methylation Predictor & Causal Model")
st.write("Dataset preview:")
st.dataframe(df.head())

# -----------------------------
# 2. Define columns
# -----------------------------
dmr_col = "location"             # DMR identifier
feature_col = "pfas_exposure"  # PFAS column
target_col = "meth.diff"       # Methylation difference column

for col in [dmr_col, feature_col, target_col]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in dataset.")
        st.stop()

# -----------------------------
# 3. Select a DMR
# -----------------------------
st.subheader("Select a DMR")
dmr_selected = st.selectbox("Choose DMR", df[dmr_col].unique())
df_dmr = df[df[dmr_col] == dmr_selected][[feature_col, target_col]].dropna()

if df_dmr.shape[0] < 2:
    st.warning("Not enough data to train a model for this DMR.")
    st.stop()

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
# 5. Causal / perturbation model
# -----------------------------
st.subheader(f"Causal / perturbation analysis for {dmr_selected}")
st.write("""
Estimate the causal effect of PFAS on methylation using a simple linear backdoor adjustment. 
This shows how changes in PFAS exposure might causally affect methylation at the selected DMR.
""")

# Simple causal model using DoWhy
try:
    causal_model = CausalModel(
        data=df_dmr,
        treatment=feature_col,
        outcome=target_col,
        common_causes=[]  # add confounders if available
    )

    identified_estimand = causal_model.identify_effect()
    estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    st.write("Estimated causal effect of PFAS on methylation:", estimate.value)

    st.subheader("Simulate methylation perturbation")
    perturb = st.slider("Change methylation (delta)", -0.5, 0.5, 0.0)
    predicted_effect = perturb * estimate.value
    st.write(f"Predicted change in outcome if methylation is perturbed: {predicted_effect:.4f}")

except Exception as e:
    st.error("Causal model could not be estimated. Make sure the data is sufficient.")
    st.write(e)
