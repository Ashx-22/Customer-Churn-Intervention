# Customer Churn Intervention System

Most churn projects stop at "predict who will churn." This one goes further:
**predict who will churn, how soon, and who can actually be saved by an
intervention** — then turns that into a per-customer recommended action.

## Why this is different from a standard churn classifier

| Standard project | This project |
|---|---|
| Binary churn / no-churn classifier | + **Survival analysis** (Cox PH, Kaplan-Meier) for *time-to-churn* |
| Optimizes accuracy/AUC | **Cost-sensitive threshold** optimized for expected profit |
| "Who will churn" | **Uplift modeling** — "who will churn *and* is persuadable by an offer" |
| Feature importance bar chart | SHAP explanations mapped to **specific retention actions** |
| Static notebook | Interactive **Streamlit app** with customer-level drill-down |

## Dataset

Bank customer churn dataset, 10,000 customers, 13 raw fields (credit score,
geography, gender, age, tenure, balance, products held, activity status,
estimated salary, churn flag). Source: `data/Churn_Modelling.csv`.

> Note on uplift modeling: this dataset has no recorded retention-offer
> experiment, so `src/uplift_simulation.py` simulates one (random treatment
> assignment + a heterogeneous treatment effect) on top of the real customer
> features. This is clearly labeled in the code and is a standard way to
> demonstrate causal/uplift methodology without real A/B test data — swap in
> a real experiment's `treatment`/`outcome` columns and the same modeling
> code runs unchanged.

## Project structure

```
churn-project/
├── data/
│   └── Churn_Modelling.csv
├── src/
│   ├── preprocessing.py         # cleaning, feature engineering, train/test split
│   ├── survival_analysis.py     # Kaplan-Meier + Cox PH time-to-churn
│   ├── classifier.py            # GBM classifier + cost-sensitive threshold
│   ├── uplift_simulation.py     # simulated retention-offer uplift model (Qini)
│   └── explainability.py        # SHAP + action recommendations
├── outputs/
│   ├── figures/                 # all generated plots
│   ├── models/                  # saved .pkl models
│   ├── processed.csv
│   ├── cox_hazard_ratios.csv
│   ├── simulated_experiment.csv
│   └── top_risk_customers_actions.csv
├── app.py                       # Streamlit dashboard
├── run_pipeline.py              # runs every step in order
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python run_pipeline.py        # runs preprocessing → survival → classifier → uplift → SHAP
streamlit run app.py          # launches the interactive dashboard
```

Every step was run end-to-end and verified error-free before delivery.

## Results (this run)

- **Classifier**: ROC-AUC ≈ 0.87
- **Cost-sensitive threshold**: moving from the default 0.50 to the
  profit-optimized threshold flips expected profit from **−$43,050** to
  **+$60,550** on the test set (business cost assumptions are set at the top
  of `classifier.py` — tune to your own numbers)
- **Survival analysis**: number of products and account activity are the
  strongest predictors of *how fast* a customer churns, not just *whether*
  they do (see `outputs/cox_hazard_ratios.csv`)
- **Uplift model**: Qini AUC ≈ 0.23 — meaningfully better than random
  targeting, meaning the model finds customers who are genuinely persuadable
  by a retention offer rather than just high-risk in general

## How to present this in a portfolio / on LinkedIn

Don't lead with "built a churn model with X% accuracy." Lead with the
business framing:

> "Built a churn *intervention* system that identifies not just which
> customers are likely to leave, but which of them can actually be retained
> by an offer — using survival analysis for time-to-churn, uplift modeling
> for targeting, and a cost-sensitive decision layer that turned a
> loss-making default threshold into a profitable one."

That sentence does the differentiation work a confusion matrix screenshot
can't.
