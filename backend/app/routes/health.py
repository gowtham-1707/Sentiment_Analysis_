
import os
import platform
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from loguru import logger

from app.schemas.request import HealthResponse, ReadyResponse

try:
    import psutil
except ImportError:
    psutil = None

router = APIRouter()
API_START_TIME = time.time()


def _get_uptime_seconds() -> float:
    return round(time.time() - API_START_TIME, 2)


def _get_system_metrics() -> dict:
    if psutil is None:
        logger.warning("psutil is not installed; system metrics are unavailable.")
        return {
            "cpu_percent":    -1,
            "memory_percent": -1,
            "disk_percent":   -1,
        }

    try:
        return {
            "cpu_percent":    psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent":   psutil.disk_usage("/").percent,
        }
    except Exception as e:
        logger.warning(f"Could not fetch system metrics: {e}")
        return {
            "cpu_percent":    -1,
            "memory_percent": -1,
            "disk_percent":   -1,
        }

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description=(
        "Returns the health status of the API. "
        "Returns HTTP 200 if healthy, HTTP 503 if the model is not loaded. "
        "Used by Docker HEALTHCHECK and load balancers."
    ),
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "API is unhealthy — model not loaded"},
    },
)
async def health_check(request: Request):
    model_service = getattr(request.app.state, "model_service", None)
    model_loaded  = model_service is not None and model_service.is_loaded

    status_code = 200 if model_loaded else 503
    status_text = "healthy" if model_loaded else "unhealthy"

    logger.debug(f"Health check | status={status_text} | model_loaded={model_loaded}")

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content=HealthResponse(
            status=status_text,
            version="1.0.0",
            model_loaded=model_loaded,
        ).model_dump(),
    )

@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness Check",
    description=(
        "Returns readiness status of the API. "
        "Checks both model availability and MLflow connectivity. "
        "Returns HTTP 200 only when fully ready to serve traffic."
    ),
    responses={
        200: {"description": "API is ready to serve traffic"},
        503: {"description": "API is not ready yet"},
    },
)
async def readiness_check(request: Request):
    model_service    = getattr(request.app.state, "model_service", None)
    model_loaded     = model_service is not None and model_service.is_loaded
    mlflow_connected = False

    if model_service is not None:
        try:
            mlflow_connected = model_service.is_mlflow_reachable()
        except Exception as e:
            logger.warning(f"MLflow reachability check failed: {e}")
            mlflow_connected = False

    is_ready    = model_loaded  
    status_code = 200 if is_ready else 503
    status_text = "ready" if is_ready else "not_ready"

    logger.debug(
        f"Readiness check | status={status_text} | "
        f"model_loaded={model_loaded} | "
        f"mlflow_connected={mlflow_connected}"
    )

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content=ReadyResponse(
            status=status_text,
            model_loaded=model_loaded,
            mlflow_connected=mlflow_connected,
        ).model_dump(),
    )

@router.get(
    "/info",
    summary="API Info",
    description="Returns detailed metadata about the API, model, and system.",
)
async def api_info(request: Request):
    """
    Detailed API info endpoint.

    Returns:
        - API version and uptime
        - Model name, version, and stage
        - System resource metrics (CPU, memory, disk)
        - Runtime environment info
    """
    model_service = getattr(request.app.state, "model_service", None)
    model_info    = (
        model_service.get_model_info()
        if model_service is not None
        else {"status": "not loaded"}
    )

    system_metrics = _get_system_metrics()

    return {
        "api": {
            "name":        "Sentiment Analysis API",
            "version":     "1.0.0",
            "status":      "running",
            "uptime_seconds": _get_uptime_seconds(),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "docs_url":    "/docs",
            "metrics_url": "/metrics",
        },
        "model": model_info,
        "system": {
            "python_version": platform.python_version(),
            "platform":       platform.system(),
            "cpu_percent":    system_metrics["cpu_percent"],
            "memory_percent": system_metrics["memory_percent"],
            "disk_percent":   system_metrics["disk_percent"],
        },
        "environment": {
            "mlflow_uri":   os.getenv("MLFLOW_TRACKING_URI", "not set"),
            "model_name":   os.getenv("MODEL_NAME", "not set"),
            "model_stage":  os.getenv("MODEL_STAGE", "not set"),
            "log_level":    os.getenv("LOG_LEVEL", "INFO"),
        },
    }

@router.get(
    "/ping",
    summary="Ping",
    description="Lightweight liveness check. Returns 'pong' immediately.",
)
async def ping():

    return {
        "status":    "pong",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
