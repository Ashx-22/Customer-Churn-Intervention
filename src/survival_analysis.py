"""
02 - Survival analysis: time-to-churn instead of plain yes/no classification.
Uses Tenure (years with bank) as duration, Churn as the event indicator.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import os

OUT_DIR = "outputs"


def load_processed():
    return pd.read_csv(f"{OUT_DIR}/processed.csv")


def kaplan_meier_by_group(df, group_col, title, fname):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))
    for value, grp in df.groupby(group_col):
        kmf.fit(grp["Tenure"] + 0.5, event_observed=grp["Churn"], label=f"{group_col}={value}")
        kmf.plot_survival_function(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Tenure (years)")
    ax.set_ylabel("Survival probability (retained)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/{fname}.png", dpi=120)
    plt.close(fig)


def fit_cox_model(df):
    cph_df = df.copy()
    cph_df["duration"] = cph_df["Tenure"] + 0.5  # avoid zero-duration ties
    cph_df = cph_df.drop(columns=["Tenure"])

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cph_df, duration_col="duration", event_col="Churn")
    return cph


def main():
    df = load_processed()

    # Kaplan-Meier curves for a couple of interpretable segments
    kaplan_meier_by_group(df, "IsActiveMember", "Retention by Activity Status", "km_active_member")
    kaplan_meier_by_group(df, "NumOfProducts", "Retention by Number of Products", "km_num_products")

    # Cox Proportional Hazards model -> which factors speed up / slow down churn
    cph = fit_cox_model(df)
    print("\n=== Cox Proportional Hazards summary ===")
    summary = cph.summary[["coef", "exp(coef)", "p"]].sort_values("p")
    print(summary)

    summary.to_csv(f"{OUT_DIR}/cox_hazard_ratios.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    cph.plot(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/cox_hazard_ratios.png", dpi=120)
    plt.close(fig)

    print("\nSaved: km_active_member.png, km_num_products.png, cox_hazard_ratios.png/.csv")


if __name__ == "__main__":
    main()
