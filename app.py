"""Risk Alert Classifier - Streamlit app.
Run:  streamlit run app.py"""
import json, os
import joblib, numpy as np, pandas as pd
import streamlit as st
from utils import build_features, FEATURE_COLUMNS, CATEGORIES

st.set_page_config(page_title="Risk Alert Classifier", page_icon="🚨", layout="wide")

@st.cache_resource
def load():
    return joblib.load("model/risk_model.pkl"), joblib.load("model/model_meta.pkl")

model, meta = load()
THR_DEFAULT = float(meta["threshold"])

def predict(raw: pd.DataFrame, thr: float):
    X = build_features(raw)[FEATURE_COLUMNS]
    p = model.predict_proba(X)[:, 1]
    return p, (p >= thr).astype(int)

st.title("🚨 Risk Alert Classifier")
st.caption("Early-warning system that flags high-risk digital banking customers.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📂 Batch Prediction", "📊 Model Report"])

# ---------------- Tab 1 ----------------
with tab1:
    thr = st.slider("Alert threshold (lower = catch more risky customers, more false alarms)",
                    0.05, 0.95, THR_DEFAULT, 0.01, key="thr1")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profile")
        age = st.number_input("Age", 18, 100, 35)
        gender = st.selectbox("Gender", CATEGORIES["gender"])
        region = st.selectbox("Region", CATEGORIES["region"])
        emp = st.selectbox("Employment type", CATEGORIES["employment_type"])
        income = st.number_input("Annual income (INR)", 0, 10_000_000, 40000, 1000)
        tenure = st.number_input("Account tenure (months)", 0, 600, 36)
    with c2:
        st.subheader("Credit behaviour")
        score = st.number_input("Credit score", 300, 900, 650)
        util = st.slider("Credit utilization ratio", 0.0, 1.5, 0.35, 0.01)
        missed = st.number_input("Missed payments (12m)", 0, 20, 0)
        late = st.number_input("Avg late payment days", 0.0, 120.0, 2.0, 0.5)
        debt = st.number_input("Debt balance (INR)", 0, 10_000_000, 30000, 1000)
        cash = st.number_input("Cash advances (6m)", 0, 20, 0)
    with c3:
        st.subheader("Activity")
        txn = st.number_input("Monthly transaction count", 0, 500, 40)
        spend = st.number_input("Monthly spend (INR)", 0, 5_000_000, 20000, 500)
        comp = st.number_input("Complaints (6m)", 0, 20, 0)
        fails = st.number_input("Failed login attempts (3m)", 0, 50, 1)
        last = st.date_input("Last transaction date", pd.Timestamp("2025-11-15"))

    if st.button("Predict risk", type="primary"):
        row = pd.DataFrame([{
            "age": age, "gender": gender, "region": region, "employment_type": emp,
            "annual_income_inr": income, "credit_score": score, "credit_utilization_ratio": util,
            "missed_payments_12m": missed, "avg_late_payment_days": late,
            "monthly_transaction_count": txn, "monthly_spend_inr": spend,
            "cash_advance_count_6m": cash, "complaints_last_6m": comp,
            "failed_login_attempts_3m": fails, "account_tenure_months": tenure,
            "last_transaction_date": str(last), "debt_balance_inr": debt}])
        p, lab = predict(row, thr)
        st.progress(float(min(p[0], 1.0)), text=f"Risk probability: {p[0]*100:.1f}%  (threshold {thr*100:.0f}%)")
        if lab[0] == 1:
            st.error("🚨 HIGH RISK - flag this customer for review.")
        else:
            st.success("✅ LOW RISK - no alert.")

# ---------------- Tab 2 ----------------
with tab2:
    st.write("Upload a CSV with the same columns as the training data (`risk_status` and `customer_id` optional).")
    thr2 = st.slider("Alert threshold", 0.05, 0.95, THR_DEFAULT, 0.01, key="thr2")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        needed = [c for c in ["age","gender","region","employment_type","annual_income_inr","credit_score",
                  "credit_utilization_ratio","missed_payments_12m","avg_late_payment_days",
                  "monthly_transaction_count","monthly_spend_inr","cash_advance_count_6m",
                  "complaints_last_6m","failed_login_attempts_3m","account_tenure_months",
                  "last_transaction_date","debt_balance_inr"] if c not in raw.columns]
        if needed:
            st.error(f"Missing columns: {needed}")
        else:
            p, lab = predict(raw, thr2)
            out = raw.copy(); out["risk_probability"] = p.round(4); out["predicted_risk"] = lab
            m1, m2, m3 = st.columns(3)
            m1.metric("Customers", len(out)); m2.metric("Flagged high-risk", int(lab.sum()))
            m3.metric("Flag rate", f"{lab.mean()*100:.1f}%")
            st.dataframe(out.sort_values("risk_probability", ascending=False), width='stretch')
            st.download_button("Download predictions", out.to_csv(index=False).encode(),
                               "risk_predictions.csv", "text/csv")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader(f"Final model: {meta['model_name']}")
    mt = meta["metrics"]
    a, b, c, d, e = st.columns(5)
    a.metric("Recall (High)", f"{mt['recall']:.2f}"); b.metric("Precision", f"{mt['precision']:.2f}")
    c.metric("F1", f"{mt['f1']:.2f}"); d.metric("AUC-ROC", f"{mt['auc_roc']:.3f}")
    e.metric("Threshold", f"{mt['threshold']:.2f}")
    st.caption(f"Test set: {mt['FN_typeII']} missed risky customers (Type-II / false negatives), "
               f"{mt['FP_typeI']} false alarms (Type-I / false positives).")
    if os.path.exists("output/08_all_model_metrics.csv"):
        st.markdown("**All models compared**")
        st.dataframe(pd.read_csv("output/08_all_model_metrics.csv"), width='stretch')
    for img, cap in [("output/09_roc_all_models.png", "ROC curves"),
                     ("output/13_cm_final_model.png", "Final confusion matrix"),
                     ("output/14_feature_importance.png", "Feature importance")]:
        if os.path.exists(img):
            st.image(img, caption=cap, width=600)
