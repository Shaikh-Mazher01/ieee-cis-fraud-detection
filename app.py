import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pathlib

st.set_page_config(
    page_title="IEEE-CIS Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    # Assumes model and sample data are stored in models/ and data/
    model_path = pathlib.Path("models/fraud_model.pkl")
    sample_path = pathlib.Path("data/test_sample_200.parquet")
    
    model = joblib.load(model_path) if model_path.exists() else None
    sample_df = pd.read_parquet(sample_path) if sample_path.exists() else None
    return model, sample_df

model, df_sample = load_artifacts()

st.title("🛡️ IEEE-CIS Fraud Detection Engine")
st.markdown("""
Interactive demo for the Fraud Detection system. Select a real historical test transaction below, 
inspect or modify its features, and evaluate the model's risk score and operational decision.
""")

if model is None or df_sample is None:
    st.error("Model artifacts or test sample parquet not found. Please ensure `models/fraud_model.pkl` and `data/test_sample_200.parquet` exist.")
else:
    # Sidebar selection
    st.sidebar.header("Transaction Selector")
    sample_idx = st.sidebar.selectbox(
        "Choose a Test Transaction", 
        options=range(min(50, len(df_sample))),
        format_func=lambda i: f"Transaction #{i} (Amt: ${df_sample.iloc[i].get('TransactionAmt', 0):.2f})"
    )
    
    row = df_sample.iloc[sample_idx].copy()
    
    # Extract target if present
    true_label = row.pop('isFraud', None)
    
    st.subheader("Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        tx_amt = st.number_input("Transaction Amount ($)", value=float(row.get('TransactionAmt', 50.0)), step=10.0)
    with col2:
        product_cd = st.text_input("Product Code", value=str(row.get('ProductCD', 'W')))
    with col3:
        card1 = st.number_input("Card 1 ID", value=int(row.get('card1', 10000)))

    row['TransactionAmt'] = tx_amt
    row['ProductCD'] = product_cd
    row['card1'] = card1

    if st.button("Evaluate Risk", type="primary"):
        # Prepare feature vector matching model expectations
        X_input = pd.DataFrame([row])
        
        # Predict probability
        prob = model.predict_proba(X_input)[:, 1][0]
        threshold = 0.5 # Default or loaded threshold
        decision = "REVIEW / FLAG" if prob >= threshold else "APPROVE"
        
        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Fraud Probability", f"{prob:.2%}")
        res_col2.metric("Model Decision", decision)
        if true_label is not None:
            res_col3.metric("Actual Label", "Fraud" if true_label == 1 else "Legitimate")
            
        if prob >= threshold:
            st.warning("⚠️ Transaction flagged for manual fraud analyst review.")
        else:
            st.success("✅ Transaction processed successfully.")