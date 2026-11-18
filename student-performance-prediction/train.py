from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data import (
    PROJECT_ROOT, FEATURES, TARGET, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, load_dataset
)

MODEL_DIR = PROJECT_ROOT / "models"


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=4, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": rmse,
            "R2": r2_score(y_test, predictions),
        })
        fitted[name] = pipeline

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    results.to_csv(MODEL_DIR / "model_metrics.csv", index=False)

    print(f"\nDataset rows used: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("\nModel comparison (lower MAE/RMSE is better; higher R² is better):")
    print(results.to_string(index=False, formatters={
        "MAE": "{:.3f}".format, "RMSE": "{:.3f}".format, "R2": "{:.3f}".format
    }))
    print(f"\nBest model by test RMSE: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved metrics: {MODEL_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
