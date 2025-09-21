import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Optional: install XGBoost if you want
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# -----------------------------
# CONFIG
# -----------------------------
MLFLOW_PORT = 5001
mlflow.set_tracking_uri(f"http://127.0.0.1:{MLFLOW_PORT}")
mlflow.set_experiment("ML_Experiment_Manager")

# Datasets to run
DATASETS = {
    "iris": load_iris(as_frame=True),
    "wine": load_wine(as_frame=True),
    "breast_cancer": load_breast_cancer(as_frame=True)
}

# Models and hyperparameters
MODEL_CONFIGS = {
    "RandomForest": [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": None}
    ]
}

if xgb_available:
    MODEL_CONFIGS["XGBoost"] = [
        {"n_estimators": 50, "max_depth": 3, "use_label_encoder": False, "eval_metric": "logloss"},
        {"n_estimators": 100, "max_depth": 5, "use_label_encoder": False, "eval_metric": "logloss"}
    ]

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(X, y, dataset_name, model_type, hyperparams):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "RandomForest":
        clf = RandomForestClassifier(**hyperparams)
    elif model_type == "XGBoost" and xgb_available:
        clf = XGBClassifier(**hyperparams)
    else:
        raise ValueError(f"Model {model_type} not supported or XGBoost not installed.")

    run_name = f"{model_type}_{dataset_name}_{hyperparams.get('n_estimators', 'default')}"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)

        # Train model
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        # Log metrics
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        print(f"[{run_name}] Accuracy: {acc}")

        # Log model
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            input_example=X_train.head(5)
        )

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Starting fully automatic ML Experiment Manager...\n")
    for dataset_name, dataset in DATASETS.items():
        X, y = dataset.data, dataset.target
        print(f"Running experiments on dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")

        for model_type, hyperparam_sets in MODEL_CONFIGS.items():
            for params in hyperparam_sets:
                train_model(X, y, dataset_name, model_type, params)

    print("\nAll experiments completed. Check MLflow UI for results!")