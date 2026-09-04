"""
03 - Churn classifier + cost-sensitive threshold optimization.
Instead of optimizing accuracy/AUC alone, we pick the probability threshold
that maximizes expected profit given business costs.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"

# Business assumptions (tune to taste / mention as assumptions in the writeup)
COST_FALSE_POSITIVE = 50     # cost of a retention offer given to a non-churner
COST_FALSE_NEGATIVE = 500    # lost revenue when a churner is missed
BENEFIT_TRUE_POSITIVE = 300  # net value of retaining a customer we correctly targeted


def load_splits():
    return joblib.load(f"{OUT_DIR}/splits.pkl")


def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def expected_profit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    profit = tp * BENEFIT_TRUE_POSITIVE - fp * COST_FALSE_POSITIVE - fn * COST_FALSE_NEGATIVE
    return profit


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.95, 0.01)
    profits = [expected_profit(y_true, y_prob, t) for t in thresholds]
    best_idx = int(np.argmax(profits))
    return thresholds[best_idx], profits[best_idx], thresholds, profits


def main():
    X_train, X_test, y_train, y_test = load_splits()

    model = train_model(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"Test ROC-AUC: {auc:.4f}")

    best_t, best_profit, thresholds, profits = find_best_threshold(y_test.values, y_prob)
    default_profit = expected_profit(y_test.values, y_prob, 0.5)

    print(f"\nDefault threshold 0.50 -> expected profit: {default_profit}")
    print(f"Optimized threshold {best_t:.2f} -> expected profit: {best_profit}")
    print(f"Profit improvement from cost-sensitive threshold: {best_profit - default_profit}")

    y_pred_best = (y_prob >= best_t).astype(int)
    print("\nClassification report at optimized threshold:")
    print(classification_report(y_test, y_pred_best))

    # Save profit-vs-threshold curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, profits)
    ax.axvline(best_t, color="red", linestyle="--", label=f"Optimal threshold = {best_t:.2f}")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Expected profit ($)")
    ax.set_title("Cost-Sensitive Threshold Optimization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/threshold_profit_curve.png", dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/roc_curve.png", dpi=120)
    plt.close(fig)

    joblib.dump(model, f"{OUT_DIR}/models/churn_classifier.pkl")
    joblib.dump(best_t, f"{OUT_DIR}/models/best_threshold.pkl")
    print("\nSaved: churn_classifier.pkl, best_threshold.pkl, threshold_profit_curve.png, roc_curve.png")


if __name__ == "__main__":
    main()
