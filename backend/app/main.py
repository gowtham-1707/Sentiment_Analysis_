
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette_exporter import PrometheusMiddleware, handle_metrics

from app.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    app_info,
)
from app.routes import health
from app.routes import predict
from app.services.model import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Sentiment Analysis API...")
    logger.info("Loading model from MLflow registry...")

    try:
        model_service = ModelService()
        model_service.load_model()

        app.state.model_service = model_service
        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model — /predict will return 503")
        app.state.model_service = None

    app_info.labels(version="1.0.0", environment="production").set(1)

    logger.info("Sentiment Analysis API is ready!")

    yield  
    logger.info("Shutting down Sentiment Analysis API...")
    app.state.model_service = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="""
    ## Product Review Sentiment Analysis

    An end-to-end MLOps-powered API that classifies product reviews
    as **Positive**, **Negative**, or **Neutral**.

    ### Features
    - Single review prediction
    - Bulk CSV prediction
    - Model health monitoring
    - Prometheus metrics at `/metrics`

    ### MLOps Stack
    - **Model Tracking:** MLflow
    - **Pipeline:** Apache Airflow
    - **Monitoring:** Prometheus + Grafana
    - **Versioning:** Git + DVC
    """,
    version="1.0.0",
    docs_url="/docs",        
    redoc_url="/redoc",      
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   
        "http://frontend:3000",    
        "*",                      
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    PrometheusMiddleware,
    app_name="sentiment_api",
    prefix="sentiment",
    group_paths=True,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        status_code = 500
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    latency = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(latency)

    # Log the request
    logger.info(
        f"{request.method} {request.url.path} "
        f"| status={status_code} "
        f"| latency={latency:.4f}s "
        f"| client={request.client.host if request.client else 'unknown'}"
    )

    return response

app.include_router(
    health.router,
    tags=["Health"],
)

app.include_router(
    predict.router,
    prefix="/api/v1",
    tags=["Predictions"],
)

app.add_route("/metrics", handle_metrics)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Sentiment Analysis API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.error(f"Unhandled exception at {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred.",
            "path": str(request.url.path),
            "error": str(exc),
        },
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       
        log_level="info",
    )
