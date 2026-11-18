# Student Performance Prediction (Machine Learning)

A complete, beginner-friendly ML project that predicts a student's **final Portuguese course grade (G3, 0–20)** using the UCI Student Performance dataset.

## Dataset (real-world)
- Source: UCI Machine Learning Repository — Student Performance
- URL: https://archive.ics.uci.edu/dataset/320/student+performance
- File used by default: `student-por.csv` (Portuguese language course; 649 student records)
- The project downloads the original UCI ZIP automatically the first time it runs and stores the CSV in `data/`.
- The dataset is observational and comes from two Portuguese secondary schools. It is not an Indian-school dataset and should not be treated as representative of every school.

## Features used
School, sex, age, address, study time, past class failures, school/family support, internet access, absences, and earlier-period grades (G1 and G2). `G3` is the target.

G1 and G2 are included because they are earlier-period grades. For a prediction made before the school year begins, remove them from `FEATURES` in `src/data.py` and retrain.

## Project structure
```text
student-performance-prediction/
├── app.py                  # Streamlit prediction dashboard
├── train.py                # Train, compare, and save models
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── data.py             # Dataset download, preparation, feature list
├── data/                   # Created automatically; original CSV is downloaded here
└── models/                 # Created by train.py
```

## Run on Windows / macOS / Linux
Python 3.10 or newer is recommended.

1. Extract the ZIP and open a terminal in the project folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train and compare models:
   ```bash
   python train.py
   ```
5. Start the web app:
   ```bash
   streamlit run app.py
   ```

The first run needs an internet connection to download the dataset from UCI. Later runs use the local copy.

## Models and evaluation
The training script compares:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

It reports **MAE**, **RMSE**, and **R²** on a held-out test set. The model with the lowest test RMSE is saved as `models/best_model.joblib`; metrics are saved to `models/model_metrics.csv`.

- MAE/RMSE are grade-point errors; lower is better.
- R² measures how much variation in test grades is explained by the model; closer to 1 is generally better.
- A fixed random seed makes the split reproducible.

## Important limitations and responsible use
- This is an educational prediction tool, not an official grading system.
- Predictions are estimates, not guarantees. Do not use them to label, rank, punish, or deny opportunities to students.
- The dataset is limited in size, geography, time, and school context. Results may not generalize to other countries or schools.
- Some inputs (such as family support or internet access) are sensitive contextual factors. Use them thoughtfully and avoid treating associations as causes.
- The model predicts an observed grade; it does not measure a student's ability or potential.

## Suggested viva / presentation points
1. Why is this a regression problem? Because G3 is a numeric grade.
2. Why split the data? To estimate performance on unseen records.
3. Why use a preprocessing pipeline? It handles numeric and categorical columns consistently and prevents preprocessing leakage.
4. Why compare several algorithms? Different models capture different patterns.
5. What is overfitting? When a model learns training-specific patterns and performs poorly on new data.
6. What is data leakage? Using information unavailable at the intended prediction time. G1/G2 are valid only when those earlier grades are already known.
