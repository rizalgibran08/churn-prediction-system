import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import dagshub
dagshub.init(repo_owner='rizalgibran08',
             repo_name='telco-churn-mlflow', mlflow=True)

# Load data
data = pd.read_csv("telco_preprocessed.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Churn", axis=1),
    data["Churn"],
    random_state=42,
    test_size=0.2
)
input_example = X_train.iloc[:5]

# Hyperparameter grid
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range = np.linspace(1, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

mlflow.set_experiment("Telco Churn RF Manual Tuning")

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"RF_{n_estimators}_{max_depth}"):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Logging manual
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )

            # Track best
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {"n_estimators": n_estimators,
                               "max_depth": max_depth}

print(f"Best accuracy: {best_accuracy}")
print(f"Best parameters: {best_params}")
