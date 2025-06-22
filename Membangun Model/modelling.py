import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Atur MLflow lokal tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco Churn")

# Load data hasil preprocessing
df = pd.read_csv("Telco_preprocessed.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Churn", axis=1),
    df["Churn"],
    test_size=0.2,
    random_state=42
)

input_example = X_train[0:5]

# Mulai run MLflow
with mlflow.start_run(run_name="Random Forest Autolog"):
    mlflow.sklearn.autolog()
    # Train model
    model = RandomForestClassifier(random_state=42)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
