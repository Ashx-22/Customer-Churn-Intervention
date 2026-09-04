"""
05 - SHAP explainability: turn model output into per-customer,
business-readable retention actions instead of a bare feature-importance plot.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"

# Simple rule book mapping top SHAP driver -> suggested action
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


def recommend_action(top_feature):
    return ACTION_MAP.get(top_feature, DEFAULT_ACTION)


def main():
    X_train, X_test, y_train, y_test = joblib.load(f"{OUT_DIR}/splits.pkl")
    model = joblib.load(f"{OUT_DIR}/models/churn_classifier.pkl")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global summary plot
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Per-customer action table for the top-20 highest-risk customers in the test set
    probs = model.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(probs)[::-1][:20]

    records = []
    for i in top_idx:
        row_shap = shap_values[i]
        top_feature = X_test.columns[np.argmax(np.abs(row_shap))]
        records.append({
            "customer_row": X_test.index[i],
            "churn_probability": round(probs[i], 3),
            "top_driver": top_feature,
            "recommended_action": recommend_action(top_feature),
        })

    action_table = pd.DataFrame(records)
    action_table.to_csv(f"{OUT_DIR}/top_risk_customers_actions.csv", index=False)
    print(action_table.to_string(index=False))

    print("\nSaved: shap_summary.png, top_risk_customers_actions.csv")


if __name__ == "__main__":
    main()
