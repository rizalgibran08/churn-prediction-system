from prometheus_client import start_http_server, Summary, Gauge, Counter
import random
import time
import psutil


# MODEL PERFORMANCE METRICS
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the model')
MODEL_PRECISION = Gauge('model_precision', 'Precision of the model')
MODEL_RECALL = Gauge('model_recall', 'Recall of the model')
MODEL_F1_SCORE = Gauge('model_f1_score', 'F1 Score of the model')
MODEL_VERSION = Gauge('model_version_info',
                      'Currently used model version (misalnya, 1 = v1)')

# REQUEST METRICS
REQUEST_LATENCY = Summary('request_latency_seconds',
                          'Latency of inference requests')
REQUEST_COUNT = Gauge('inference_requests_total',
                      'Total number of inference requests')
FAILED_REQUESTS = Counter('failed_requests_total',
                          'Total number of failed inference requests')

# SYSTEM METRICS
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent',
                     'System memory usage percentage')

# OPERATIONAL METRICS
UPTIME = Counter('uptime_seconds_total', 'Uptime of exporter in seconds')


def simulate_metrics():
    while True:
        # Model Performance Metrics
        MODEL_ACCURACY.set(random.uniform(0.75, 0.95))
        MODEL_PRECISION.set(random.uniform(0.75, 0.95))
        MODEL_RECALL.set(random.uniform(0.70, 0.95))
        MODEL_F1_SCORE.set(random.uniform(0.70, 0.95))
        MODEL_VERSION.set(1)

        # Request Metrics
        latency = random.uniform(0.1, 0.6)
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.inc()
        if random.random() < 0.1:  # 10% kemungkinan gagal
            FAILED_REQUESTS.inc()

        # System Metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        # Operational Metrics
        UPTIME.inc(5)

        time.sleep(5)


if __name__ == '__main__':
    print("Exporter Prometheus berjalan di port 8000...")
    start_http_server(8000)
    simulate_metrics()
