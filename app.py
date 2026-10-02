import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(page_title="IEEE-CIS Fraud Detection", page_icon="🛡️", layout="centered")

st.title("🛡️ AI Fraud Detection System")
st.write("Enter transaction details below to check if it is **APPROVE** or **REVIEW**.")

# Load the trained model bundle safely
@st.cache_resource
def load_model():
    model_path = Path("outputs/real/models/bundle.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None

bundle = load_model()

if bundle is None:
    st.error("Model bundle not found! Make sure outputs/real/models/bundle.joblib is pushed to GitHub.")
else:
    model = bundle["model"]
    builder = bundle["builder"]
    threshold = bundle.get("threshold", 0.02)

    # Input form
    st.subheader("Transaction Parameters")
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=117.50)
    product_cd = st.selectbox("Product Code (ProductCD)", ["W", "C", "R", "H", "S"])
    card4 = st.selectbox("Card Type (card4)", ["visa", "mastercard", "discover", "amex"])
    email = st.text_input("Email Domain (P_emaildomain)", "anonymous.com")

    if st.button("Analyze Transaction"):
        # Create a mock dataframe matching input schema
        input_data = pd.DataFrame([{
            "TransactionAmt": amt,
            "ProductCD": product_cd,
            "card4": card4,
            "P_emaildomain": email,
            "TransactionDT": 86400  # Default dummy time
        }])
        
        # Transform features and predict
        try:
            X_trans = builder.transform(input_data)
            prob = float(model.predict_proba(X_trans)[:, 1][0])
            
            st.divider()
            st.metric(label="Fraud Probability Score", value=f"{prob:.4f}")
            
            if prob >= threshold:
                st.error("🚨 **Decision: REVIEW** (High Risk of Fraud)")
            else:
                st.success("✅ **Decision: APPROVE** (Low Risk)")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")