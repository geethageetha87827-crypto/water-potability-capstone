# Water Quality Risk Classification (Capstone Problem 09)

## Problem
Classify water-sample observations into potability categories (safe / not safe to drink)
using measurable water-quality attributes.

## Dataset
- **Name:** Water Potability
- **Source:** Kaggle — https://www.kaggle.com/datasets/adityakadiwal/water-potability
- **Size:** 3,276 rows, 9 numeric features + binary target (`Potability`)
- **License/citation:** Please cite the Kaggle dataset page in your technical paper's References section.
- Download `water_potability.csv` and place it in `data/water_potability.csv` before running the notebook.

## Project structure
```
.
├── data/
│   └── water_potability.csv      # you add this (not included — see Dataset above)
├── train_model.ipynb              # EDA, preprocessing, model training & evaluation
├── app.py                         # Streamlit prototype app
├── requirements.txt
└── README.md
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place `water_potability.csv` inside `data/`.
3. Open and run `train_model.ipynb` top to bottom (Jupyter or VS Code).
   This will save `model.joblib`, `scaler.joblib`, `medians.joblib`,
   `feature_order.joblib`, and `model_name.joblib` into the project folder.
4. Launch the prototype app:
   ```
   streamlit run app.py
   ```
5. Enter sample values in the browser UI and click **Predict potability**.

## Models tried
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree

The notebook trains all three, compares them on Accuracy, Precision, Recall, F1 and ROC-AUC,
and automatically selects the best model by F1-score for the saved artifacts (you can override
this manually in the notebook if you have a good reason, e.g. prioritizing recall for a safety task).

## Evaluation metrics
Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC — see notebook output and
your technical paper for the write-up and interpretation.

## Limitations
- Small dataset (3,276 rows) — results may not generalize well.
- Missing values were imputed with the column median, which is a simplification.
- This is an educational prototype, not a validated tool for real water-safety decisions.

## Future scope
- Try ensemble methods (Random Forest, Gradient Boosting) beyond the foundational-model scope.
- Collect more data / additional contextual features (location, season).
- Explore more principled imputation and outlier handling.

## Author / Credits
- Dataset: Aditya Kadiwal, Kaggle.
- Capstone brief: Learn Depth Academy LLP, Track 1, Problem 09.
