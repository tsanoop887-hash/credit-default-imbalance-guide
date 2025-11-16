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
