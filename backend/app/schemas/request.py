from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator



class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL  = "Neutral"


class ModelStage(str, Enum):
    PRODUCTION  = "Production"
    STAGING     = "Staging"
    ARCHIVED    = "Archived"

class SingleReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The product review text to classify",
        examples=["This product is absolutely amazing! Works perfectly."],
    )
    product_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional product ID for tracking purposes",
        examples=["B001234"],
    )

    @field_validator("review")
    @classmethod
    def review_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Review text cannot be blank or whitespace only.")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "review": "This product is absolutely amazing! Works perfectly.",
                    "product_id": "B001234",
                }
            ]
        }
    }


class BulkReviewRequest(BaseModel):
    
    reviews: List[SingleReviewRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of reviews to classify (max 500 per request)",
    )

    @field_validator("reviews")
    @classmethod
    def reviews_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("At least one review must be provided.")
        return v

class SentimentResult(BaseModel):
    review: str = Field(
        ...,
        description="The original review text"
    )
    sentiment: SentimentLabel = Field(
        ...,
        description="Predicted sentiment label"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
        examples=[0.92],
    )
    probabilities: dict = Field(
        ...,
        description="Probability score for each sentiment class",
        examples=[{"Positive": 0.92, "Negative": 0.05, "Neutral": 0.03}],
    )
    product_id: Optional[str] = Field(
        default=None,
        description="Product ID if provided in request"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "review": "This product is absolutely amazing!",
                    "sentiment": "Positive",
                    "confidence": 0.92,
                    "probabilities": {
                        "Positive": 0.92,
                        "Negative": 0.05,
                        "Neutral": 0.03,
                    },
                    "product_id": "B001234",
                }
            ]
        }
    }


class SinglePredictionResponse(BaseModel):
    status: str = Field(default="success", description="Request status")
    data: SentimentResult = Field(..., description="Prediction result")
    model_version: str = Field(
        ...,
        description="MLflow model version used for prediction",
        examples=["3"],
    )
    inference_time_ms: float = Field(
        ...,
        description="Time taken for inference in milliseconds",
        examples=[12.5],
    )


class BulkPredictionResponse(BaseModel):
    status: str = Field(default="success", description="Request status")
    total: int = Field(..., description="Total number of reviews processed")
    results: List[SentimentResult] = Field(
        ...,
        description="List of prediction results"
    )
    summary: dict = Field(
        ...,
        description="Sentiment distribution summary",
        examples=[{"Positive": 10, "Negative": 3, "Neutral": 2}],
    )
    model_version: str = Field(
        ...,
        description="MLflow model version used for prediction"
    )
    inference_time_ms: float = Field(
        ...,
        description="Total inference time in milliseconds"
    )

class HealthResponse(BaseModel):
    status: str = Field(
        ...,
        description="Health status of the API",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready"
    )

class ReadyResponse(BaseModel):
    status: str = Field(
        ...,
        description="Readiness status",
        examples=["ready"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded"
    )
    mlflow_connected: bool = Field(
        ...,
        description="Whether MLflow tracking server is reachable"
    )

class ErrorResponse(BaseModel):
    status: str = Field(default="error", description="Error status")
    detail: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Review text cannot be blank."],
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code",
        examples=["INVALID_INPUT"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "error",
                    "detail": "Review text cannot be blank.",
                    "error_code": "INVALID_INPUT",
                }
            ]
        }
    }