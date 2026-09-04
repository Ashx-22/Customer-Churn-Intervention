"""
Runs the entire pipeline end to end, in order.
Usage: python run_pipeline.py
Then:  streamlit run app.py
"""
from src import preprocessing, survival_analysis, classifier, uplift_simulation, explainability

STEPS = [
    ("Preprocessing", preprocessing.main),
    ("Survival analysis", survival_analysis.main),
    ("Classifier + cost-sensitive threshold", classifier.main),
    ("Uplift simulation", uplift_simulation.main),
    ("SHAP explainability", explainability.main),
]

if __name__ == "__main__":
    for name, fn in STEPS:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        fn()
    print("\nPipeline complete. Run: streamlit run app.py")
