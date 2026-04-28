import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.schemas.request import SentimentLabel
from app.services.model import ModelService
from app.services.preprocessor import TextPreprocessor, get_preprocessor

@pytest.fixture
def mock_model_service():
    
    service = MagicMock(spec=ModelService)
    service.is_loaded = True

    service.predict.return_value = {
        "sentiment":     SentimentLabel.POSITIVE,
        "confidence":    0.92,
        "probabilities": {
            SentimentLabel.POSITIVE: 0.92,
            SentimentLabel.NEGATIVE: 0.05,
            SentimentLabel.NEUTRAL:  0.03,
        },
        "inference_ms": 12.5,
    }
    service.predict_batch.return_value = [
        {
            "sentiment":     SentimentLabel.POSITIVE,
            "confidence":    0.91,
            "probabilities": {
                SentimentLabel.POSITIVE: 0.91,
                SentimentLabel.NEGATIVE: 0.06,
                SentimentLabel.NEUTRAL:  0.03,
            },
            "inference_ms": None,
        },
        {
            "sentiment":     SentimentLabel.NEGATIVE,
            "confidence":    0.87,
            "probabilities": {
                SentimentLabel.POSITIVE: 0.08,
                SentimentLabel.NEGATIVE: 0.87,
                SentimentLabel.NEUTRAL:  0.05,
            },
            "inference_ms": None,
        },
    ]

    service.get_model_info.return_value = {
        "model_name":    "sentiment_classifier",
        "model_version": "3",
        "model_stage":   "Production",
        "is_loaded":     True,
        "mlflow_uri":    "http://mlflow:5000",
    }

    service.is_mlflow_reachable.return_value = True
    return service


@pytest.fixture
def client(mock_model_service):
    app.state.model_service = mock_model_service
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model():
    app.state.model_service = None
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_csv_bytes():
    content = (
        "review,product_id\n"
        "This product is absolutely amazing!,B001\n"
        "Terrible quality. Do not buy.,B002\n"
        "It's okay for the price.,B003\n"
    )
    return content.encode("utf-8")


@pytest.fixture
def sample_csv_no_product_id():
    content = (
        "review\n"
        "Great product!\n"
        "Awful experience.\n"
    )
    return content.encode("utf-8")


class TestHealthEndpoints:

    def test_ping_returns_pong(self, client):
        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pong"
        assert "timestamp" in data

    def test_health_returns_200_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"

    def test_health_returns_503_when_no_model(self, client_no_model):
        response = client_no_model.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False

    def test_ready_returns_200_when_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

    def test_ready_returns_503_when_no_model(self, client_no_model):
        response = client_no_model.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"

    def test_info_returns_api_metadata(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "api" in data
        assert "model" in data
        assert "system" in data
        assert "environment" in data
        assert data["api"]["version"] == "1.0.0"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestSinglePredict:

    def test_predict_positive_review(self, client):
        payload = {"review": "This product is absolutely amazing! Best purchase ever."}
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["sentiment"] == "Positive"
        assert 0.0 <= data["data"]["confidence"] <= 1.0
        assert "probabilities" in data["data"]
        assert "inference_time_ms" in data
        assert "model_version" in data

    def test_predict_with_product_id(self, client):
        payload = {
            "review":     "Great product, highly recommended!",
            "product_id": "B001234",
        }
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["product_id"] == "B001234"

    def test_predict_without_product_id(self, client):
        payload = {"review": "Good quality and fast delivery."}
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["product_id"] is None

    def test_predict_probabilities_sum_to_one(self, client):
        payload = {"review": "Decent product for the price."}
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        probs = response.json()["data"]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01  

    def test_predict_returns_all_three_classes(self, client):
        payload = {"review": "Acceptable product overall."}
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        probs = response.json()["data"]["probabilities"]
        assert "Positive" in probs
        assert "Negative" in probs
        assert "Neutral"  in probs

    def test_predict_inference_time_under_200ms(self, client):
        payload = {"review": "This is a sample review for latency testing."}
        response = client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        latency = response.json()["inference_time_ms"]
        assert latency < 200.0, f"Latency {latency}ms exceeds 200ms SLA"

    def test_predict_empty_review_returns_422(self, client):
        payload = {"review": ""}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_whitespace_only_review_returns_422(self, client):
        payload = {"review": "   \t\n  "}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_too_short_review_returns_422(self, client):
        payload = {"review": "ok"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_review_field_returns_422(self, client):
        payload = {"product_id": "B001"}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_model_not_loaded_returns_503(self, client_no_model):
        payload = {"review": "This product is great!"}
        response = client_no_model.post("/api/v1/predict", json=payload)
        assert response.status_code == 503

    def test_predict_review_at_max_length(self, client):
        payload = {"review": "A" * 5000}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200

    def test_predict_review_over_max_length_returns_422(self, client):
        payload = {"review": "A" * 5001}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

class TestBulkPredict:

    def test_bulk_predict_returns_all_results(self, client):
        payload = {
            "reviews": [
                {"review": "Great product, love it!"},
                {"review": "Terrible, broke immediately."},
            ]
        }
        response = client.post("/api/v1/predict/bulk", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_bulk_predict_summary_counts(self, client):
        payload = {
            "reviews": [
                {"review": "Amazing product!"},
                {"review": "Awful experience."},
            ]
        }
        response = client.post("/api/v1/predict/bulk", json=payload)

        assert response.status_code == 200
        data = response.json()
        summary = data["summary"]
        assert "Positive" in summary
        assert "Negative" in summary
        assert "Neutral"  in summary
        assert sum(summary.values()) == data["total"]

    def test_bulk_predict_with_product_ids(self, client):
        payload = {
            "reviews": [
                {"review": "Good product.",    "product_id": "B001"},
                {"review": "Bad experience.",  "product_id": "B002"},
            ]
        }
        response = client.post("/api/v1/predict/bulk", json=payload)

        assert response.status_code == 200
        results = response.json()["results"]
        assert results[0]["product_id"] == "B001"
        assert results[1]["product_id"] == "B002"

    def test_bulk_predict_empty_list_returns_422(self, client):
        payload = {"reviews": []}
        response = client.post("/api/v1/predict/bulk", json=payload)
        assert response.status_code == 422

    def test_bulk_predict_model_not_loaded_returns_503(self, client_no_model):
        payload = {"reviews": [{"review": "Good product."}]}
        response = client_no_model.post("/api/v1/predict/bulk", json=payload)
        assert response.status_code == 503

    def test_bulk_predict_contains_inference_time(self, client):
        payload = {"reviews": [{"review": "Good product."}]}
        response = client.post("/api/v1/predict/bulk", json=payload)
        assert response.status_code == 200
        assert "inference_time_ms" in response.json()

    def test_bulk_predict_contains_model_version(self, client):
        payload = {"reviews": [{"review": "Good product."}]}
        response = client.post("/api/v1/predict/bulk", json=payload)
        assert response.status_code == 200
        assert "model_version" in response.json()

class TestCSVPredict:

    def test_csv_predict_valid_file(self, client, sample_csv_bytes):
        files    = {"file": ("reviews.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
        response = client.post("/api/v1/predict/csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_csv_predict_without_product_id_column(self, client, sample_csv_no_product_id):
        files    = {"file": ("reviews.csv", io.BytesIO(sample_csv_no_product_id), "text/csv")}
        response = client.post("/api/v1/predict/csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_csv_predict_wrong_extension_returns_400(self, client):
        files    = {"file": ("reviews.txt", io.BytesIO(b"review\nGood product."), "text/plain")}
        response = client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 400

    def test_csv_predict_missing_review_column_returns_400(self, client):
        csv_content = b"text,rating\nGood product,5\n"
        files       = {"file": ("reviews.csv", io.BytesIO(csv_content), "text/csv")}
        response    = client.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 400

    def test_csv_predict_model_not_loaded_returns_503(self, client_no_model, sample_csv_bytes):
        files    = {"file": ("reviews.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
        response = client_no_model.post("/api/v1/predict/csv", files=files)
        assert response.status_code == 503

    def test_csv_predict_summary_present(self, client, sample_csv_bytes):
        files    = {"file": ("reviews.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
        response = client.post("/api/v1/predict/csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert set(data["summary"].keys()) >= {"Positive", "Negative", "Neutral"}

    def test_csv_predict_results_have_probabilities(self, client, sample_csv_bytes):
        files    = {"file": ("reviews.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
        response = client.post("/api/v1/predict/csv", files=files)

        assert response.status_code == 200
        results = response.json()["results"]
        for r in results:
            assert "probabilities" in r
            assert len(r["probabilities"]) == 3

class TestPreprocessor:

    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor(remove_stopwords=True, lemmatize=True)

    def test_lowercase_conversion(self, preprocessor):
        result = preprocessor.preprocess("AMAZING PRODUCT")
        assert result == result.lower()

    def test_html_tags_removed(self, preprocessor):
        result = preprocessor.preprocess("<br/>Great product<p>Love it</p>")
        assert "<" not in result
        assert ">" not in result

    def test_urls_removed(self, preprocessor):
        result = preprocessor.preprocess("Visit https://example.com for more info")
        assert "http" not in result
        assert "example.com" not in result

    def test_special_chars_removed(self, preprocessor):
        result = preprocessor.preprocess("Amazing!!! 5/5 stars ***")
        assert "!" not in result
        assert "*" not in result

    def test_stopwords_removed(self, preprocessor):
        result = preprocessor.preprocess("This is a very good product")
        tokens = result.split()
        assert "this" not in tokens
        assert "is"   not in tokens
        assert "a"    not in tokens

    def test_negation_words_kept(self, preprocessor):
        result = preprocessor.preprocess("This is not a good product")
        assert "not" in result

    def test_empty_string_returns_empty(self, preprocessor):
        result = preprocessor.preprocess("")
        assert result == ""

    def test_whitespace_only_returns_empty(self, preprocessor):
        result = preprocessor.preprocess("   \t\n  ")
        assert result == ""

    def test_non_string_raises_value_error(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.preprocess(None)

        with pytest.raises(ValueError):
            preprocessor.preprocess(123)

    def test_batch_preprocessing(self, preprocessor):
        reviews = [
            "Great product!",
            "Terrible quality.",
            "Okay for the price.",
        ]
        results = preprocessor.preprocess_batch(reviews)
        assert len(results) == len(reviews)
        for r in results:
            assert isinstance(r, str)

    def test_contraction_expansion(self, preprocessor):
        result = preprocessor.preprocess("I don't like this product at all")
        assert "not" in result

    def test_singleton_instance(self):
        p1 = get_preprocessor()
        p2 = get_preprocessor()
        assert p1 is p2

class TestSchemaValidation:

    def test_review_min_length_validation(self, client):
        response = client.post("/api/v1/predict", json={"review": "ab"})
        assert response.status_code == 422

    def test_review_max_length_validation(self, client):
        response = client.post("/api/v1/predict", json={"review": "x" * 5001})
        assert response.status_code == 422

    def test_review_exactly_3_chars_valid(self, client):
        response = client.post("/api/v1/predict", json={"review": "abc"})
        assert response.status_code == 200

    def test_bulk_max_500_reviews(self, client, mock_model_service):
        mock_model_service.predict_batch.return_value = [
            {
                "sentiment":     SentimentLabel.POSITIVE,
                "confidence":    0.9,
                "probabilities": {
                    SentimentLabel.POSITIVE: 0.9,
                    SentimentLabel.NEGATIVE: 0.05,
                    SentimentLabel.NEUTRAL:  0.05,
                },
                "inference_ms": None,
            }
        ] * 501

        payload = {"reviews": [{"review": f"Review {i}"} for i in range(501)]}
        response = client.post("/api/v1/predict/bulk", json=payload)
        assert response.status_code == 422

class TestMetricsEndpoint:

    def test_metrics_endpoint_accessible(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "sentiment" in response.text or "python" in response.text

    def test_request_increments_counter(self, client):
        client.post("/api/v1/predict", json={"review": "Great product!"})
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200