import csv
import io
import time
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from loguru import logger

from app.monitoring.metrics import (
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    BULK_REQUEST_SIZE,
)
from app.schemas.request import (
    BulkPredictionResponse,
    BulkReviewRequest,
    ErrorResponse,
    SentimentResult,
    SinglePredictionResponse,
    SingleReviewRequest,
)
from app.services.model import ModelService


router = APIRouter()

def get_model(request: Request) -> ModelService:
    model_service = getattr(request.app.state, "model_service", None)

    if model_service is None or not model_service.is_loaded:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status":     "error",
                "detail":     "Model is not loaded yet. Please try again shortly.",
                "error_code": "MODEL_NOT_READY",
            },
        )
    return model_service


@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict sentiment for a single review",
    description=(
        "Accepts a single product review text and returns the predicted "
        "sentiment label (Positive / Negative / Neutral) along with "
        "confidence scores and per-class probabilities."
    ),
    responses={
        200: {"description": "Prediction successful"},
        422: {"description": "Validation error — invalid input"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_single(
    body: SingleReviewRequest,
    model_service: ModelService = Depends(get_model),
) -> SinglePredictionResponse:
    logger.info(
        f"Single predict request | "
        f"product_id={body.product_id} | "
        f"review_length={len(body.review)}"
    )

    try:
        result = model_service.predict(body.review)
        PREDICTION_COUNTER.labels(
            sentiment=result["sentiment"].value,
            endpoint="single",
        ).inc()
        PREDICTION_LATENCY.labels(endpoint="single").observe(
            result["inference_ms"] / 1000
        )

        sentiment_result = SentimentResult(
            review=body.review,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            product_id=body.product_id,
        )

        model_info = model_service.get_model_info()

        logger.info(
            f"Prediction complete | "
            f"sentiment={result['sentiment'].value} | "
            f"confidence={result['confidence']:.2%} | "
            f"inference={result['inference_ms']}ms"
        )

        return SinglePredictionResponse(
            status="success",
            data=sentiment_result,
            model_version=model_info["model_version"],
            inference_time_ms=result["inference_ms"],
        )

    except ValueError as e:
        logger.warning(f"Validation error in predict_single: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status":     "error",
                "detail":     str(e),
                "error_code": "INVALID_INPUT",
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error in predict_single: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status":     "error",
                "detail":     "An unexpected error occurred during inference.",
                "error_code": "INFERENCE_ERROR",
            },
        )

@router.post(
    "/predict/bulk",
    response_model=BulkPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict sentiment for multiple reviews (JSON)",
    description=(
        "Accepts a JSON list of up to 500 product reviews and returns "
        "sentiment predictions for all of them in a single response. "
        "Also returns a sentiment distribution summary."
    ),
    responses={
        200: {"description": "Bulk prediction successful"},
        422: {"description": "Validation error — invalid input"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_bulk(
    body: BulkReviewRequest,
    model_service: ModelService = Depends(get_model),
) -> BulkPredictionResponse:
    total_reviews = len(body.reviews)
    logger.info(f"Bulk predict request | count={total_reviews}")
    BULK_REQUEST_SIZE.observe(total_reviews)

    start_time = time.time()

    try:
        review_texts = [r.review for r in body.reviews]
        predictions = model_service.predict_batch(review_texts)

        results: List[SentimentResult] = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for i, (review_req, pred) in enumerate(zip(body.reviews, predictions)):
            sentiment_label = pred["sentiment"].value

            sentiment_counts[sentiment_label] = (
                sentiment_counts.get(sentiment_label, 0) + 1
            )

            PREDICTION_COUNTER.labels(
                sentiment=sentiment_label,
                endpoint="bulk",
            ).inc()

            results.append(
                SentimentResult(
                    review=review_req.review,
                    sentiment=pred["sentiment"],
                    confidence=pred["confidence"],
                    probabilities=pred["probabilities"],
                    product_id=review_req.product_id,
                )
            )

        total_ms   = round((time.time() - start_time) * 1000, 2)
        model_info = model_service.get_model_info()
        PREDICTION_LATENCY.labels(endpoint="bulk").observe(total_ms / 1000)

        logger.info(
            f"Bulk prediction complete | "
            f"total={total_reviews} | "
            f"time={total_ms}ms | "
            f"distribution={sentiment_counts}"
        )

        return BulkPredictionResponse(
            status="success",
            total=total_reviews,
            results=results,
            summary=sentiment_counts,
            model_version=model_info["model_version"],
            inference_time_ms=total_ms,
        )

    except Exception as e:
        logger.error(f"Unexpected error in predict_bulk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status":     "error",
                "detail":     "Bulk inference failed unexpectedly.",
                "error_code": "BULK_INFERENCE_ERROR",
            },
        )

@router.post(
    "/predict/csv",
    status_code=status.HTTP_200_OK,
    summary="Predict sentiment from uploaded CSV file",
    description=(
        "Accepts a CSV file upload where each row contains a product review. "
        "The CSV must have a column named 'review'. "
        "An optional 'product_id' column is also supported. "
        "Returns predictions for all rows as JSON."
    ),
    responses={
        200: {"description": "CSV prediction successful"},
        400: {"description": "Invalid CSV format"},
        413: {"description": "File too large (max 5MB)"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_csv(
    file: UploadFile = File(
        ...,
        description="CSV file with a 'review' column (max 5MB, max 500 rows)"
    ),
    model_service: ModelService = Depends(get_model),
):
    
    MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  
    MAX_ROWS            = 500

    logger.info(
        f"CSV predict request | "
        f"filename={file.filename} | "
        f"content_type={file.content_type}"
    )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status":     "error",
                "detail":     "Only .csv files are accepted.",
                "error_code": "INVALID_FILE_TYPE",
            },
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status":     "error",
                "detail":     f"Could not read uploaded file: {e}",
                "error_code": "FILE_READ_ERROR",
            },
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "status":     "error",
                "detail":     f"File too large. Maximum allowed size is 5MB.",
                "error_code": "FILE_TOO_LARGE",
            },
        )

    try:
        decoded     = contents.decode("utf-8")
        reader      = csv.DictReader(io.StringIO(decoded))
        rows        = list(reader)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status":     "error",
                "detail":     "CSV file must be UTF-8 encoded.",
                "error_code": "ENCODING_ERROR",
            },
        )
    if not rows or "review" not in rows[0]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status":     "error",
                "detail":     "CSV must contain a 'review' column.",
                "error_code": "MISSING_COLUMN",
            },
        )
    if len(rows) > MAX_ROWS:
        logger.warning(
            f"CSV has {len(rows)} rows. Truncating to {MAX_ROWS}."
        )
        rows = rows[:MAX_ROWS]
    review_texts = []
    product_ids  = []
    for row in rows:
        review_text = row.get("review", "").strip()
        if review_text:
            review_texts.append(review_text)
            product_ids.append(row.get("product_id", None))

    if not review_texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status":     "error",
                "detail":     "No valid reviews found in uploaded CSV.",
                "error_code": "EMPTY_CSV",
            },
        )

    logger.info(f"Parsed {len(review_texts)} valid reviews from CSV.")
    start_time = time.time()

    try:
        predictions     = model_service.predict_batch(review_texts)
        results         = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

        for review_text, product_id, pred in zip(
            review_texts, product_ids, predictions
        ):
            sentiment_label = pred["sentiment"].value
            sentiment_counts[sentiment_label] = (
                sentiment_counts.get(sentiment_label, 0) + 1
            )

            PREDICTION_COUNTER.labels(
                sentiment=sentiment_label,
                endpoint="csv",
            ).inc()

            results.append(
                SentimentResult(
                    review=review_text,
                    sentiment=pred["sentiment"],
                    confidence=pred["confidence"],
                    probabilities=pred["probabilities"],
                    product_id=product_id,
                )
            )

        total_ms   = round((time.time() - start_time) * 1000, 2)
        model_info = model_service.get_model_info()

        PREDICTION_LATENCY.labels(endpoint="csv").observe(total_ms / 1000)
        BULK_REQUEST_SIZE.observe(len(review_texts))

        logger.info(
            f"CSV prediction complete | "
            f"total={len(review_texts)} | "
            f"time={total_ms}ms | "
            f"distribution={sentiment_counts}"
        )

        return BulkPredictionResponse(
            status="success",
            total=len(review_texts),
            results=results,
            summary=sentiment_counts,
            model_version=model_info["model_version"],
            inference_time_ms=total_ms,
        )

    except Exception as e:
        logger.error(f"CSV inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status":     "error",
                "detail":     "CSV inference failed unexpectedly.",
                "error_code": "CSV_INFERENCE_ERROR",
            },
        )