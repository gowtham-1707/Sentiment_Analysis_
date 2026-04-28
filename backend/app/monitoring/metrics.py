from prometheus_client import Counter, Gauge, Histogram, Info, Summary

app_info = Gauge(
    name="sentiment_app_info",
    documentation="Sentiment Analysis API metadata",
    labelnames=["version", "environment"],
)

REQUEST_COUNT = Counter(
    name="sentiment_request_total",
    documentation=(
        "Total number of HTTP requests received. "
        "Labels: method (GET/POST), endpoint (/predict etc.), status_code"
    ),
    labelnames=["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    name="sentiment_request_latency_seconds",
    documentation=(
        "HTTP request latency in seconds. "
        "Labels: method, endpoint"
    ),
    labelnames=["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
)

REQUESTS_IN_PROGRESS = Gauge(
    name="sentiment_custom_requests_in_progress",
    documentation="Number of HTTP requests currently being processed.",
    labelnames=["endpoint"],
)

PREDICTION_COUNTER = Counter(
    name="sentiment_predictions_total",
    documentation=(
        "Total number of sentiment predictions made. "
        "Labels: sentiment (Positive/Negative/Neutral), endpoint (single/bulk/csv)"
    ),
    labelnames=["sentiment", "endpoint"],
)

PREDICTION_LATENCY = Histogram(
    name="sentiment_prediction_latency_seconds",
    documentation=(
        "ML inference latency in seconds (preprocessing + vectorization + classify). "
        "Labels: endpoint (single/bulk/csv)"
    ),
    labelnames=["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1.0],
)

PREDICTION_CONFIDENCE = Histogram(
    name="sentiment_prediction_confidence",
    documentation=(
        "Distribution of model confidence scores (0.0 to 1.0). "
        "Low confidence may indicate data drift."
    ),
    labelnames=["sentiment"],
    buckets=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)

PREDICTION_ERROR_COUNTER = Counter(
    name="sentiment_prediction_errors_total",
    documentation=(
        "Total number of prediction errors. "
        "Labels: error_code (INVALID_INPUT, INFERENCE_ERROR etc.)"
    ),
    labelnames=["error_code", "endpoint"],
)

BULK_REQUEST_SIZE = Histogram(
    name="sentiment_bulk_request_size",
    documentation="Number of reviews per bulk/CSV prediction request.",
    buckets=[1, 5, 10, 25, 50, 100, 200, 300, 500],
)


MODEL_LOAD_STATUS = Gauge(
    name="sentiment_model_loaded",
    documentation=(
        "Whether the ML model is currently loaded. "
        "1 = loaded, 0 = not loaded."
    ),
)

MODEL_VERSION_INFO = Gauge(
    name="sentiment_model_version_info",
    documentation="Current model version loaded from MLflow registry.",
    labelnames=["model_name", "version", "stage"],
)

MLFLOW_CONNECTED = Gauge(
    name="sentiment_mlflow_connected",
    documentation=(
        "Whether the MLflow tracking server is reachable. "
        "1 = connected, 0 = disconnected."
    ),
)

DATA_DRIFT_SCORE = Gauge(
    name="sentiment_data_drift_score",
    documentation=(
        "Estimated data drift score (0.0 = no drift, 1.0 = max drift). "
        "Computed by comparing incoming review feature distributions "
        "against training baseline statistics."
    ),
    labelnames=["feature"],
)

SENTIMENT_DISTRIBUTION = Gauge(
    name="sentiment_label_distribution_ratio",
    documentation=(
        "Rolling ratio of each sentiment label in recent predictions. "
        "Alerts if distribution shifts significantly from training baseline."
    ),
    labelnames=["sentiment"],
)

for label in ["Positive", "Negative", "Neutral"]:
    SENTIMENT_DISTRIBUTION.labels(sentiment=label).set(0)

ACTIVE_WORKERS = Gauge(
    name="sentiment_active_workers",
    documentation="Number of active Uvicorn worker processes.",
)

PREPROCESSING_LATENCY = Summary(
    name="sentiment_preprocessing_latency_seconds",
    documentation=(
        "Time spent in text preprocessing per review "
        "(lowercase + clean + tokenize + lemmatize)."
    ),
)

VECTORIZATION_LATENCY = Summary(
    name="sentiment_vectorization_latency_seconds",
    documentation="Time spent in TF-IDF vectorization per batch.",
)

CSV_UPLOAD_COUNTER = Counter(
    name="sentiment_csv_uploads_total",
    documentation="Total number of CSV file uploads received.",
    labelnames=["status"],  
)

CSV_UPLOAD_SIZE_BYTES = Histogram(
    name="sentiment_csv_upload_size_bytes",
    documentation="Size of uploaded CSV files in bytes.",
    buckets=[
        1024,           
        10240,          
        102400,         
        512000,         
        1048576,        
        2097152,        
        5242880,        
    ],
)


def record_prediction(
    sentiment: str,
    confidence: float,
    endpoint: str,
    inference_seconds: float,
) -> None:
    """
    Record a single prediction across all relevant metrics.

    Args:
        sentiment:         Predicted label (Positive/Negative/Neutral)
        confidence:        Model confidence score (0.0 - 1.0)
        endpoint:          Which endpoint served the request (single/bulk/csv)
        inference_seconds: Inference time in seconds
    """
    PREDICTION_COUNTER.labels(
        sentiment=sentiment,
        endpoint=endpoint,
    ).inc()

    PREDICTION_CONFIDENCE.labels(
        sentiment=sentiment,
    ).observe(confidence)

    PREDICTION_LATENCY.labels(
        endpoint=endpoint,
    ).observe(inference_seconds)


def record_prediction_error(error_code: str, endpoint: str) -> None:
    """
    Record a prediction error in Prometheus.

    Args:
        error_code: Machine-readable error code (e.g. INVALID_INPUT)
        endpoint:   Which endpoint encountered the error
    """
    PREDICTION_ERROR_COUNTER.labels(
        error_code=error_code,
        endpoint=endpoint,
    ).inc()


def update_model_status(
    is_loaded: bool,
    model_name: str = "",
    version: str = "",
    stage: str = "",
    mlflow_connected: bool = False,
) -> None:
    """
    Update model health gauges in Prometheus.
    Called during model load/reload events.

    Args:
        is_loaded:        Whether model is currently loaded
        model_name:       MLflow model name
        version:          MLflow model version
        stage:            MLflow model stage (Production/Staging)
        mlflow_connected: Whether MLflow is reachable
    """
    MODEL_LOAD_STATUS.set(1 if is_loaded else 0)
    MLFLOW_CONNECTED.set(1 if mlflow_connected else 0)

    if is_loaded and model_name:
        MODEL_VERSION_INFO.labels(
            model_name=model_name,
            version=version,
            stage=stage,
        ).set(1)


def update_sentiment_distribution(counts: dict, total: int) -> None:
    if total == 0:
        return

    for label, count in counts.items():
        ratio = round(count / total, 4)
        SENTIMENT_DISTRIBUTION.labels(sentiment=label).set(ratio)
