# Credit Default Prediction

**Predicting credit-card payment default using machine learning (imbalanced classification).**

This repository contains an end-to-end project for predicting whether a credit-card client will default the next month using the UCI "Default of Credit Card Clients" dataset. The pipeline includes data preprocessing, imbalance handling (SMOTE / fallback oversampling), model training (LightGBM / XGBoost / stacking), hyperparameter tuning (Optuna / RandomizedSearch), evaluation, and export of the best model and scaler.

---

## 🔥 Highlights
- Handles severe class imbalance
- Uses LightGBM & XGBoost with Optuna tuning (if installed)
- Stacking ensemble + threshold optimization for best F1/Recall
- Exportable artifacts: `best_model.pkl`, `scaler.pkl`, `threshold.txt`
- Project report included: `project_report.md`

---

## 📁 Repository structure
credit-default-prediction/
│
├── README.md
├── project_report.md
├── requirements.txt
├── .gitignore
├── LICENSE                   # optional
│
├── data/
│   └── default_of_credit_card_clients.xls     # (ignored in git)
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Stacking_Ensemble_and_Tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── tuning.py
│   ├── ensemble.py
│   ├── threshold_opt.py
│   └── main.py                        # main pipeline runner
│
├── models/
│   ├── lgb_model.pkl
│   ├── xgb_model.pkl
│   ├── stacking_meta.pkl
│   └── scaler.pkl
│
├── artifacts/
│   ├── threshold.txt
│   ├── model_results.json
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── images/
│   ├── architecture.png
│   ├── flowchart.png
│   └── dataset_distribution.png
│
└── .github/
    └── workflows/
        └── ci.yml            # optional CI/CD workflow

