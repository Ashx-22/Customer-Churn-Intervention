"""
Streamlit app: Customer Churn Intervention System
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

st.set_page_config(page_title="Churn Intervention System", layout="wide")

OUT_DIR = "outputs"

ACTION_MAP = {
    "IsActiveMember": "Re-engagement campaign (push notifications / app onboarding nudge)",
    "NumOfProducts": "Cross-sell a second product with a fee waiver",
    "ProductsPerTenureYear": "Cross-sell a second product with a fee waiver",
    "Age": "Assign to senior-relationship-manager outreach track",
    "IsSeniorCustomer": "Assign to senior-relationship-manager outreach track",
    "Balance": "Offer a preferential savings/interest rate",
    "BalanceSalaryRatio": "Offer a preferential savings/interest rate",
    "CreditScore": "Offer a credit-score improvement consultation",
    "Geography_Germany": "Route to region-specific retention desk (Germany)",
    "EngagementScore": "Re-engagement campaign (push notifications / app onboarding nudge)",
    "ZeroBalance": "Prompt to set up direct deposit / minimum balance incentive",
}
DEFAULT_ACTION = "General retention call from customer success"


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{OUT_DIR}/models/churn_classifier.pkl")
    threshold = joblib.load(f"{OUT_DIR}/models/best_threshold.pkl")
    uplift_model = joblib.load(f"{OUT_DIR}/models/uplift_model.pkl")
    X_train, X_test, y_train, y_test = joblib.load(f"{OUT_DIR}/splits.pkl")
    processed = pd.read_csv(f"{OUT_DIR}/processed.csv")
    explainer = shap.TreeExplainer(model)
    return model, threshold, uplift_model, X_test, y_test, processed, explainer


model, threshold, uplift_model, X_test, y_test, processed, explainer = load_artifacts()

st.title(" Customer Churn Intervention System")
st.caption(
    "Not just 'who will churn' — this app predicts churn risk, time-to-churn, "
    "and who is actually persuadable by a retention offer."
)

tab1, tab2, tab3 = st.tabs(["Customer Lookup", "Portfolio Overview", "Retention Targeting"])

with tab1:
    st.subheader("Individual customer risk")
    row_choice = st.selectbox("Select a test-set customer (by row index)", X_test.index.tolist())
    row = X_test.loc[[row_choice]]

    prob = model.predict_proba(row)[0, 1]
    risk_flag = "🔴 High risk" if prob >= threshold else "🟢 Low risk"

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{prob:.1%}")
    col2.metric("Decision", risk_flag)
    col3.metric("Decision threshold (cost-optimized)", f"{threshold:.2f}")

    row_shap = explainer.shap_values(row)[0]
    top_feature = row.columns[np.argmax(np.abs(row_shap))]
    st.info(f"**Top driver:** {top_feature}  →  **Suggested action:** {ACTION_MAP.get(top_feature, DEFAULT_ACTION)}")

    uplift_score = uplift_model.predict(row)[0]
    st.write(f"**Predicted retention-offer uplift for this customer:** {uplift_score:+.3f} "
             f"(positive = offer likely reduces their churn risk)")

    st.write("SHAP feature contributions:")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.bar_plot(row_shap, feature_names=row.columns, max_display=8, show=False)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Portfolio-level churn drivers")
    c1, c2 = st.columns(2)
    with c1:
        st.image(f"{OUT_DIR}/figures/km_active_member.png", caption="Retention by activity status")
    with c2:
        st.image(f"{OUT_DIR}/figures/km_num_products.png", caption="Retention by number of products")

    c3, c4 = st.columns(2)
    with c3:
        st.image(f"{OUT_DIR}/figures/shap_summary.png", caption="Global SHAP feature importance")
    with c4:
        st.image(f"{OUT_DIR}/figures/threshold_profit_curve.png", caption="Cost-sensitive threshold optimization")

with tab3:
    st.subheader("Who to target with a retention offer")
    st.caption(
        "Ranked by predicted uplift (simulated retention-offer experiment) — "
        "high-risk customers are not always the best targets; persuadable customers are."
    )
    uplift_scores = uplift_model.predict(X_test)
    target_table = X_test.copy()
    target_table["churn_probability"] = model.predict_proba(X_test)[:, 1]
    target_table["predicted_uplift"] = uplift_scores
    target_table = target_table.sort_values("predicted_uplift", ascending=False)
    st.dataframe(
        target_table[["churn_probability", "predicted_uplift"]].head(25),
        use_container_width=True,
    )
    st.image(f"{OUT_DIR}/figures/qini_curve.png", caption="Qini curve — targeting efficiency vs. random targeting")
