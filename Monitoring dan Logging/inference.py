from flask import Flask, request, jsonify
import mlflow.pyfunc
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

model = mlflow.pyfunc.load_model("models:/telco_churn_model/Production")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        instances = data.get("instances", None)
        if instances is None:
            return jsonify({"error": "Missing 'instances' key"}), 400
        prediction = model.predict(instances).tolist()
        return jsonify({"predictions": prediction})
    return jsonify({"error": "Request must be JSON"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
