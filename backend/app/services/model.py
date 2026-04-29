import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from loguru import logger
from sklearn.pipeline import Pipeline
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.schemas.request import SentimentLabel
from app.services.preprocessor import get_preprocessor

LABEL_MAP: Dict[int, SentimentLabel] = {
    0: SentimentLabel.NEGATIVE,
    1: SentimentLabel.NEUTRAL,
    2: SentimentLabel.POSITIVE,
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "sentiment_classifier")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")
LOCAL_MODEL_PATH    = os.getenv("LOCAL_MODEL_PATH", "/app/models/sentiment_model.pkl")
LOCAL_VECTORIZER_PATH = os.getenv("LOCAL_VECTORIZER_PATH", "/app/models/tfidf_vectorizer.pkl")

class ModelService:
    def __init__(self) -> None:
        self.model          = None       
        self.vectorizer     = None       
        self.model_version  = "unknown"  
        self.model_stage    = "unknown"  
        self.preprocessor   = get_preprocessor()  
        self._is_loaded     = False      

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _load_from_mlflow(self) -> None:
        
        logger.info(
            f"Loading model '{MODEL_NAME}' "
            f"(stage='{MODEL_STAGE}') from MLflow..."
        )
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        try:
            loaded = mlflow.sklearn.load_model(model_uri)
            if isinstance(loaded, Pipeline):
                self.model      = loaded
                self.vectorizer = None  # Vectorizer is inside the pipeline
                logger.info("Loaded sklearn Pipeline (vectorizer + classifier bundled)")
            else:
                self.model = loaded
                self._load_vectorizer_artifact()
                logger.info("Loaded standalone classifier from MLflow")
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                self.model_version = versions[0].version
                self.model_stage   = versions[0].current_stage
                logger.info(
                    f"Model version: {self.model_version} | "
                    f"Stage: {self.model_stage}"
                )

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"MLflow error: {e}")
            raise

    def _load_vectorizer_artifact(self) -> None:
        try:
            client     = mlflow.tracking.MlflowClient()
            versions   = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                run_id         = versions[0].run_id
                artifact_uri   = f"runs:/{run_id}/vectorizer"
                self.vectorizer = mlflow.sklearn.load_model(artifact_uri)
                logger.info("TF-IDF vectorizer loaded from MLflow artifacts")
        except Exception as e:
            logger.warning(
                f"Could not load vectorizer from MLflow: {e}. "
                f"Trying local path..."
            )
            self._load_vectorizer_local()

    def _load_vectorizer_local(self) -> None:
        if os.path.exists(LOCAL_VECTORIZER_PATH):
            self.vectorizer = joblib.load(LOCAL_VECTORIZER_PATH)
            logger.info(f"Vectorizer loaded from local path: {LOCAL_VECTORIZER_PATH}")
        else:
            raise FileNotFoundError(
                f"Vectorizer not found at {LOCAL_VECTORIZER_PATH}"
            )

    def _load_from_local(self) -> None:
        logger.warning("Falling back to local model files...")

        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Local model not found at: {LOCAL_MODEL_PATH}. "
                f"Please train the model first using training/train.py"
            )

        self.model         = joblib.load(LOCAL_MODEL_PATH)
        self.model_version = "local"
        self.model_stage   = "local"
        logger.info(f"Model loaded from local path: {LOCAL_MODEL_PATH}")
        if isinstance(self.model, Pipeline):
            self.vectorizer = None
            logger.info("Loaded local sklearn Pipeline (vectorizer + classifier bundled)")
        else:
            self._load_vectorizer_local()

    def load_model(self) -> None:
        try:
            self._load_from_mlflow()
            self._is_loaded = True
            logger.info("Model ready for inference (via MLflow)")

        except Exception as mlflow_error:
            logger.warning(
                f"MLflow loading failed: {mlflow_error}. "
                f"Attempting local fallback..."
            )
            try:
                self._load_from_local()
                self._is_loaded = True
                logger.info("Model ready for inference (via local fallback)")

            except Exception as local_error:
                logger.error(
                    f"Both MLflow and local model loading failed.\n"
                    f"   MLflow error : {mlflow_error}\n"
                    f"   Local error  : {local_error}"
                )
                raise RuntimeError(
                    "Failed to load model from MLflow or local storage. "
                    "Please ensure the model is trained and available."
                ) from local_error

    def _vectorize(self, clean_texts: List[str]) -> np.ndarray:
    
        if isinstance(self.model, Pipeline):
            return clean_texts
        return self.vectorizer.transform(clean_texts)

    def _decode_prediction(
        self,
        predicted_class: int,
        probabilities: np.ndarray,
    ) -> Tuple[SentimentLabel, float, Dict[str, float]]:
        label      = LABEL_MAP.get(predicted_class, SentimentLabel.NEUTRAL)
        confidence = float(round(max(probabilities), 4))

        proba_dict = {
            SentimentLabel.NEGATIVE: float(round(probabilities[0], 4)),
            SentimentLabel.NEUTRAL:  float(round(probabilities[1], 4)),
            SentimentLabel.POSITIVE: float(round(probabilities[2], 4)),
        }

        return label, confidence, proba_dict

    def predict(self, review_text: str) -> Dict:
        if not self._is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() first."
            )

        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty.")

        start_time = time.time()
        clean_text = self.preprocessor.preprocess(review_text)

        if not clean_text:
            logger.warning(
                f"Review reduced to empty string after preprocessing. "
                f"Defaulting to Neutral."
            )
            return {
                "sentiment":     SentimentLabel.NEUTRAL,
                "confidence":    0.5,
                "probabilities": {
                    SentimentLabel.NEGATIVE: 0.25,
                    SentimentLabel.NEUTRAL:  0.50,
                    SentimentLabel.POSITIVE: 0.25,
                },
                "inference_ms":  0.0,
            }

        features = self._vectorize([clean_text])
        if isinstance(self.model, Pipeline):
            predicted_class  = int(self.model.predict([clean_text])[0])
            probabilities    = self.model.predict_proba([clean_text])[0]
        else:
            predicted_class  = int(self.model.predict(features)[0])
            probabilities    = self.model.predict_proba(features)[0]
        label, confidence, proba_dict = self._decode_prediction(
            predicted_class, probabilities
        )

        inference_ms = round((time.time() - start_time) * 1000, 2)

        logger.debug(
            f"Prediction: {label} ({confidence:.2%}) | "
            f"inference={inference_ms}ms"
        )

        return {
            "sentiment":     label,
            "confidence":    confidence,
            "probabilities": proba_dict,
            "inference_ms":  inference_ms,
        }

    def predict_batch(self, review_texts: List[str]) -> List[Dict]:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        logger.info(f"Running batch inference on {len(review_texts)} reviews...")
        start_time = time.time()
        clean_texts = self.preprocessor.preprocess_batch(review_texts)
        if isinstance(self.model, Pipeline):
            predicted_classes = self.model.predict(clean_texts)
            all_probabilities = self.model.predict_proba(clean_texts)
        else:
            features          = self._vectorize(clean_texts)
            predicted_classes = self.model.predict(features)
            all_probabilities = self.model.predict_proba(features)

        results = []
        for pred_class, probas in zip(predicted_classes, all_probabilities):
            label, confidence, proba_dict = self._decode_prediction(
                int(pred_class), probas
            )
            results.append({
                "sentiment":     label,
                "confidence":    confidence,
                "probabilities": proba_dict,
                "inference_ms":  None,  
            })

        total_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"Batch inference complete | "
            f"total={total_ms}ms | "
            f"avg={round(total_ms / max(len(review_texts), 1), 2)}ms/review"
        )

        return results

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_model_info(self) -> Dict:
        return {
            "model_name":    MODEL_NAME,
            "model_version": self.model_version,
            "model_stage":   self.model_stage,
            "is_loaded":     self._is_loaded,
            "mlflow_uri":    MLFLOW_TRACKING_URI,
        }

    def is_mlflow_reachable(self) -> bool:
        
        try:
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()
            return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    
    service = ModelService()
    service.load_model()
    return service
