
import os
import pandas as pd
import joblib
import mlflow

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import xgboost as xgb


# --------------------------------------------------
# 1. Load training and testing datasets
# --------------------------------------------------

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")

ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")


# CustomerID is an identifier and is not used for prediction
if "CustomerID" in Xtrain.columns:
    Xtrain = Xtrain.drop(columns=["CustomerID"])
    Xtest = Xtest.drop(columns=["CustomerID"])


# --------------------------------------------------
# 2. Identify numerical and categorical columns
# --------------------------------------------------

categorical_columns = Xtrain.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = Xtrain.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 3. Create preprocessing pipelines
# --------------------------------------------------

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = make_column_transformer(
    (numeric_transformer, numerical_columns),
    (categorical_transformer, categorical_columns)
)


# --------------------------------------------------
# 4. Define XGBoost model
# --------------------------------------------------

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

pipeline = make_pipeline(
    preprocessor,
    model
)


# --------------------------------------------------
# 5. Define hyperparameter grid
# --------------------------------------------------

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}


# --------------------------------------------------
# 6. Hyperparameter tuning
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)


# --------------------------------------------------
# 7. MLflow experiment tracking
# --------------------------------------------------

mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(Xtest)

    accuracy = accuracy_score(ytest, predictions)
    precision = precision_score(ytest, predictions)
    recall = recall_score(ytest, predictions)
    f1 = f1_score(ytest, predictions)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print("Best Parameters:")
    print(grid_search.best_params_)

    print("\nClassification Report:")
    print(classification_report(ytest, predictions))

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------
    # 8. Save trained model
    # --------------------------------------------------

    os.makedirs("tourism_project/deployment", exist_ok=True)

    MODEL_PATH = "tourism_project/deployment/model.pkl"

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model saved successfully at: {MODEL_PATH}")
