"""
04 - Uplift modeling: "who will churn AND can be saved by a retention offer"
instead of just "who will churn".

NOTE: The real dataset has no recorded retention-offer experiment, so we
SIMULATE one on top of the real customer features. This is a standard and
honest way to demonstrate uplift/causal methodology in a portfolio project
when no A/B test data exists — it is clearly labeled as simulated, and the
same code runs unchanged on a real experiment's data if you get access to one.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklift.models import SoloModel
from sklift.metrics import qini_auc_score, qini_curve

OUT_DIR = "outputs"
RNG = np.random.default_rng(42)


def load_processed():
    return pd.read_csv(f"{OUT_DIR}/processed.csv")


def simulate_experiment(df):
    df = df.copy()
    base_model = joblib.load(f"{OUT_DIR}/models/churn_classifier.pkl")
    feature_cols = [c for c in df.columns if c != "Churn"]

    p0 = base_model.predict_proba(df[feature_cols])[:, 1]  # baseline churn propensity

    # Randomly assign the retention offer (as in a real A/B test)
    treatment = RNG.binomial(1, 0.5, size=len(df))

    # Heterogeneous treatment effect (simulated):
    # the offer works best on inactive, single-product, higher-balance customers
    # and does ~nothing (or mildly backfires) on already-loyal, multi-product customers
    effect = (
        0.18 * (df["IsActiveMember"] == 0)
        + 0.12 * (df["NumOfProducts"] == 1)
        + 0.05 * (df["Balance"] > df["Balance"].median())
        - 0.05 * (df["NumOfProducts"] >= 3)
    )
    p1 = np.clip(p0 - effect, 0.01, 0.99)

    p_effective = np.where(treatment == 1, p1, p0)
    observed_churn = RNG.binomial(1, p_effective)

    df["treatment"] = treatment
    df["observed_churn"] = observed_churn
    df["true_uplift"] = p0 - p1  # ground truth, for validation only
    return df, feature_cols


def main():
    df, feature_cols = simulate_experiment(load_processed())

    X = df[feature_cols]
    trt = df["treatment"]
    y = df["observed_churn"]

    X_train, X_test, trt_train, trt_test, y_train, y_test = train_test_split(
        X, trt, y, test_size=0.3, random_state=42, stratify=trt
    )

    uplift_model = SoloModel(estimator=GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    ))
    uplift_model.fit(X_train, y_train, trt_train)

    uplift_preds = uplift_model.predict(X_test)

    qini = qini_auc_score(y_test, uplift_preds, trt_test)
    print(f"Qini AUC score: {qini:.4f}  (>0 means the model finds real persuadable customers)")

    x_qini, y_qini = qini_curve(y_test, uplift_preds, trt_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_qini, y_qini, label="Model")
    ax.plot([x_qini[0], x_qini[-1]], [y_qini[0], y_qini[-1]], linestyle="--", color="gray", label="Random targeting")
    ax.set_xlabel("Number of customers targeted")
    ax.set_ylabel("Cumulative uplift (churns avoided)")
    ax.set_title("Qini Curve — Retention Offer Targeting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/qini_curve.png", dpi=120)
    plt.close(fig)

    # Rank customers by predicted uplift -> who to target with the retention offer
    results = X_test.copy()
    results["predicted_uplift"] = uplift_preds
    top_targets = results.sort_values("predicted_uplift", ascending=False).head(10)
    print("\nTop 10 customers to target with retention offer (highest predicted uplift):")
    print(top_targets[["predicted_uplift"]])

    joblib.dump(uplift_model, f"{OUT_DIR}/models/uplift_model.pkl")
    df.to_csv(f"{OUT_DIR}/simulated_experiment.csv", index=False)
    print("\nSaved: uplift_model.pkl, qini_curve.png, simulated_experiment.csv")


if __name__ == "__main__":
    main()
