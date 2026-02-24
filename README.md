# AI-Powered Revenue Leakage Detection & Recovery System

## What is this project?

Every business loses money it doesn't know about. A discount applied twice, an invoice paid short, a billing error nobody caught — these small leaks add up to thousands or millions in lost revenue every year. This project builds an intelligent system that automatically finds those leaks, explains *why* they happened, and tells you exactly which transactions to chase first to get your money back.

This was built as a Master's research project combining two machine learning models — an Artificial Neural Network and a Gradient Boosting Machine — with a causal explainability interface powered by SHAP. The goal was not just to predict leakage but to *explain* it in a way that a business team can actually act on.

---

## What does it actually do?

You upload a CSV of your business transactions. The system scores every single row with a risk score between 0 and 1, flags the ones most likely to be leaking revenue, tells you how much money is at risk, ranks them by recovery priority, and then explains — in plain terms — *which factors* caused each transaction to be flagged.

There is also a What-If interface where you can ask questions like *"what would happen to this transaction's risk score if we reduced the discount from 28% to 5%?"* — and the system gives you a causal answer with a before-and-after comparison.

---

## Models Used

Two models were trained and combined into an ensemble:

**Artificial Neural Network (ANN)** — A deep feedforward network with 4 hidden layers (256 → 128 → 64 → 32 neurons), batch normalization, and dropout for regularization. Trained with class-weighted binary crossentropy to handle imbalanced data.

**Gradient Boosting Machine (GBM)** — A scikit-learn GradientBoostingClassifier with 300 estimators, subsampling, and depth-limited trees. Chosen as the second model because it is SHAP-compatible and provides native feature importance.

**Ensemble** — The final prediction is a weighted average of both models (ANN 45% + GBM 55%), which consistently outperforms either model individually on ROC-AUC.

---

## The Causal Interface

Most fraud and anomaly detection systems tell you *what* is suspicious but not *why*. This project adds a causal layer using SHAP (SHapley Additive exPlanations) that breaks down every single prediction into individual feature contributions. You can see exactly how much each variable — discount rate, billing errors, payment ratio, manual overrides — pushed the risk score up or down for any given transaction.

The What-If tab takes this further by letting you simulate interventions. If a business wants to know whether fixing their billing error process would reduce leakage risk, they can test that hypothesis directly in the app without touching any code.

---

## How to Run It

This project runs entirely in Google Colab. No local setup needed.

**Step 1** — Open the notebook in Google Colab

**Step 2** — Run all cells from top to bottom (Runtime → Run All)

**Step 3** — When Step 2 runs, you will be prompted to either upload your own business CSV or use the built-in synthetic dataset to try everything out

**Step 4** — The final cell launches a Gradio web app with a public shareable link. Open that link in any browser — no Colab access needed for the demo

---

## Project Structure

```
revenue-leakage-detection/
│
├── Revenue_Leakage_Detection.ipynb   ← Main notebook (run this)
├── README.md                         ← You are here
│
├── outputs/
│   ├── recovery_priority_list.csv    ← Generated after prediction
│   ├── ann_leakage_model.h5          ← Saved ANN model
│   ├── gbm_leakage_model.pkl         ← Saved GBM model
│   └── scaler.pkl                    ← Saved feature scaler
│
└── charts/                           ← Auto-generated visualizations
    ├── eda_analysis.png
    ├── ann_training.png
    ├── model_comparison.png
    ├── shap_beeswarm.png
    └── recovery_analysis.png
```

---

## What data does it need?

The system works with any business transaction CSV. It auto-detects your columns and adapts. It works best when your data includes some combination of invoice amounts, payment amounts, discount rates, billing flags, or similar financial fields — but it will run and produce results even if your data looks different.

If you just want to see it work without any data, choose option 2 at the data loading step and it generates a realistic synthetic dataset automatically.

---

## App Features

The Gradio app has four tabs:

| Tab | What it does |
|-----|-------------|
| 📂 Batch CSV Prediction | Upload a full CSV and score every transaction at once |
| 🔍 Single Transaction Check | Enter one transaction manually and get instant risk score |
| 🧪 What-If Causal Analysis | Simulate business interventions and measure their impact |
| 🧠 SHAP Explainer | Get a causal breakdown of why any transaction was flagged |

---

## Tech Stack

- **Python 3.10+** running on Google Colab
- **TensorFlow / Keras** for the ANN
- **Scikit-learn** for GBM, preprocessing, and evaluation
- **SHAP** for causal explainability
- **Gradio** for the interactive web interface
- **Plotly + Matplotlib + Seaborn** for visualizations
- **Pandas + NumPy** for data processing

---

## Results

On the synthetic dataset the ensemble model achieves a ROC-AUC above 0.95. On real business data results will vary depending on data quality and leakage patterns, but the system is designed to be useful even with messy, incomplete, or imbalanced data.

---

## Why this matters

Revenue leakage is a real problem that costs businesses between 1% and 5% of annual revenue on average according to industry estimates. Most of it goes undetected because it doesn't look like fraud — it looks like normal transactions with slightly off numbers. Traditional rule-based systems catch the obvious cases. This system catches the subtle ones and explains them clearly enough that someone without a data science background can understand and act on the results.

---

## Author

Built as a Master's degree research project. Feel free to fork, use, and build on this work. If you find it useful or have suggestions, open an issue or reach out.

---

*If you use this project in your own research, a star on the repo would be appreciated.*
