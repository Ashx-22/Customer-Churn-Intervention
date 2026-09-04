"""
01 - Data loading & preprocessing
Bank Customer Churn dataset (10,000 customers)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = "data/Churn_Modelling.csv"
OUT_DIR = "outputs"


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df):
    df = df.copy()

    # Rename for consistency
    df = df.rename(columns={
        "Num Of Products": "NumOfProducts",
        "Has Credit Card": "HasCrCard",
        "Is Active Member": "IsActiveMember",
        "Estimated Salary": "EstimatedSalary",
    })

    # Basic engineered features that a plain classifier project usually skips
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["ProductsPerTenureYear"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["IsSeniorCustomer"] = (df["Age"] >= 50).astype(int)
    df["EngagementScore"] = df["IsActiveMember"] * 2 + df["HasCrCard"] - df["ZeroBalance"]

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    drop_cols = ["CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def make_splits(df, target="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")
    print(f"Churn rate: {raw['Churn'].mean():.3f}")

    processed = engineer_features(raw)
    processed.to_csv(f"{OUT_DIR}/processed.csv", index=False)
    print(f"Processed shape: {processed.shape}")

    X_train, X_test, y_train, y_test = make_splits(processed)
    joblib.dump((X_train, X_test, y_train, y_test), f"{OUT_DIR}/splits.pkl")
    print("Saved processed.csv and splits.pkl")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()
