import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.preprocessor import get_preprocessor

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/models/sentiment_model.pkl")
LOCAL_VECTORIZER_PATH = os.getenv(
    "LOCAL_VECTORIZER_PATH",
    "/app/models/tfidf_vectorizer.pkl",
)

def load_params(params_path: str = "training/params.yaml") -> dict:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logger.info(f"Parameters loaded from {params_path}")
    return params

def load_data(params: dict) -> tuple:
    processed_path = params["data"]["processed_path"]
    logger.info(f"Loading processed data from {processed_path}...")

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            f"Please run the Airflow data pipeline first:\n"
            f"  airflow dags trigger data_ingestion_pipeline"
        )

    data = joblib.load(processed_path)

    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    logger.info(
        f"Data loaded | "
        f"train={len(X_train)} | "
        f"val={len(X_val)} | "
        f"test={len(X_test)}"
    )

    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"  {split_name} distribution: {dist}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_pipeline(params: dict) -> Pipeline:
    tfidf_params   = params["preprocessing"]["tfidf"]
    active_model   = params["model"]["active_model"]
    model_params   = params["model"][active_model]

    logger.info(f"Building pipeline | model={active_model}")
    vectorizer = TfidfVectorizer(
        max_features  = tfidf_params["max_features"],
        ngram_range   = tuple(tfidf_params["ngram_range"]),
        min_df        = tfidf_params["min_df"],
        max_df        = tfidf_params["max_df"],
        sublinear_tf  = tfidf_params["sublinear_tf"],
        analyzer      = tfidf_params["analyzer"],
        strip_accents = tfidf_params["strip_accents"],
    )
    if active_model == "logistic_regression":
        classifier = LogisticRegression(
            C             = model_params["C"],
            max_iter      = model_params["max_iter"],
            solver        = model_params["solver"],
            multi_class   = model_params["multi_class"],
            class_weight  = model_params["class_weight"],
            n_jobs        = model_params["n_jobs"],
            random_state  = model_params["random_state"],
            tol           = model_params["tol"],
        )
    elif active_model == "xgboost":
        classifier = XGBClassifier(
            n_estimators      = model_params["n_estimators"],
            max_depth         = model_params["max_depth"],
            learning_rate     = model_params["learning_rate"],
            subsample         = model_params["subsample"],
            colsample_bytree  = model_params["colsample_bytree"],
            eval_metric       = model_params["eval_metric"],
            n_jobs            = model_params["n_jobs"],
            random_state      = model_params["random_state"],
        )
    else:
        raise ValueError(f"Unknown model: {active_model}. Choose 'logistic_regression' or 'xgboost'.")

    pipeline = Pipeline([
        ("tfidf",      vectorizer),
        ("classifier", classifier),
    ])

    logger.info(f"Pipeline built: TF-IDF → {active_model}")
    return pipeline

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
) -> None:
    
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix — Sentiment Classifier", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    output_path: str,
) -> None:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    n_classes  = len(class_names)
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Sentiment Classifier (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
) -> str:
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Sentiment Analysis — Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {output_path}")
    return report


def run_cross_validation(
    pipeline: Pipeline,
    X_train: list,
    y_train: np.ndarray,
    params: dict,
) -> dict:
    
    cv_folds   = params["training"]["cv_folds"]
    cv_scoring = params["training"]["cv_scoring"]

    logger.info(f"Running {cv_folds}-fold cross validation (scoring={cv_scoring})...")

    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=skf,
        scoring=cv_scoring,
        n_jobs=-1,
    )

    cv_results = {
        "cv_mean": round(float(scores.mean()), 4),
        "cv_std":  round(float(scores.std()), 4),
    }

    logger.info(
        f"Cross validation complete | "
        f"mean={cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}"
    )
    return cv_results


def save_local_artifacts(pipeline: Pipeline) -> None:
    """Persist local fallback artifacts for the backend container."""
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, LOCAL_MODEL_PATH)
    logger.info(f"Local fallback model saved to {LOCAL_MODEL_PATH}")

    tfidf_step = pipeline.named_steps.get("tfidf")
    if tfidf_step is not None:
        os.makedirs(os.path.dirname(LOCAL_VECTORIZER_PATH), exist_ok=True)
        joblib.dump(tfidf_step, LOCAL_VECTORIZER_PATH)
        logger.info(f"Local fallback vectorizer saved to {LOCAL_VECTORIZER_PATH}")

def train(params_path: str = "training/params.yaml") -> None:
    """
    Full training pipeline with MLflow tracking.

    Steps:
        1.  Load parameters from params.yaml
        2.  Configure MLflow experiment
        3.  Load preprocessed train/val/test data
        4.  Build sklearn Pipeline (TF-IDF + Classifier)
        5.  Run cross validation
        6.  Train on full training set
        7.  Evaluate on validation set
        8.  Evaluate on test set
        9.  Generate artifacts (confusion matrix, ROC, report)
        10. Log everything to MLflow
        11. Register model to MLflow Model Registry
        12. Promote to Staging if acceptance criteria met
    """
    params = load_params(params_path)
    mlflow_uri     = os.getenv(
        "MLFLOW_TRACKING_URI",
        params["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(params["training"]["experiment_name"])

    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    logger.info(f"Experiment: {params['training']['experiment_name']}")
    with mlflow.start_run(run_name=params["training"]["run_name"]) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started | run_id={run_id}")
        mlflow.log_params({
            "dataset":          params["data"]["dataset_name"],
            "sample_size":      params["data"]["sample_size"],
            "train_ratio":      params["data"]["train_ratio"],
            "val_ratio":        params["data"]["val_ratio"],
            "test_ratio":       params["data"]["test_ratio"],
            "random_seed":      params["data"]["random_seed"],
            "remove_stopwords": params["preprocessing"]["remove_stopwords"],
            "lemmatize":        params["preprocessing"]["lemmatize"],
            "tfidf_max_features": params["preprocessing"]["tfidf"]["max_features"],
            "tfidf_ngram_range":  str(params["preprocessing"]["tfidf"]["ngram_range"]),
            "tfidf_min_df":       params["preprocessing"]["tfidf"]["min_df"],
            "tfidf_max_df":       params["preprocessing"]["tfidf"]["max_df"],
            "tfidf_sublinear_tf": params["preprocessing"]["tfidf"]["sublinear_tf"],
            "active_model":     params["model"]["active_model"],
            **{
                f"model_{k}": v
                for k, v in params["model"][params["model"]["active_model"]].items()
            },
            "cv_folds":         params["training"]["cv_folds"],
        })
        mlflow.set_tags(params["mlflow"]["tags"])
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(params)

        mlflow.log_metrics({
            "data_train_size": len(X_train),
            "data_val_size":   len(X_val),
            "data_test_size":  len(X_test),
        })
        pipeline = build_pipeline(params)
        cv_results = run_cross_validation(pipeline, X_train, y_train, params)
        mlflow.log_metrics(cv_results)

        logger.info("Training pipeline on full training set...")
        train_start = time.time()
        pipeline.fit(X_train, y_train)
        train_time  = round(time.time() - train_start, 2)

        logger.info(f"Training complete | time={train_time}s")
        mlflow.log_metric("training_time_seconds", train_time)
        save_local_artifacts(pipeline)
        logger.info("Evaluating on validation set...")
        y_val_pred  = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)

        val_metrics = {
            "val_accuracy":         round(accuracy_score(y_val, y_val_pred), 4),
            "val_f1_macro":         round(f1_score(y_val, y_val_pred, average="macro"), 4),
            "val_f1_weighted":      round(f1_score(y_val, y_val_pred, average="weighted"), 4),
            "val_precision_macro":  round(precision_score(y_val, y_val_pred, average="macro"), 4),
            "val_recall_macro":     round(recall_score(y_val, y_val_pred, average="macro"), 4),
            "val_roc_auc_ovr":      round(roc_auc_score(y_val, y_val_proba, multi_class="ovr"), 4),
        }
        mlflow.log_metrics(val_metrics)
        logger.info(f"  Validation metrics: {val_metrics}")
        logger.info("Evaluating on test set...")
        y_test_pred  = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)
        infer_start    = time.time()
        pipeline.predict(X_test[:100])
        infer_latency  = round(((time.time() - infer_start) / 100) * 1000, 2)

        test_metrics = {
            "test_accuracy":         round(accuracy_score(y_test, y_test_pred), 4),
            "test_f1_macro":         round(f1_score(y_test, y_test_pred, average="macro"), 4),
            "test_f1_weighted":      round(f1_score(y_test, y_test_pred, average="weighted"), 4),
            "test_precision_macro":  round(precision_score(y_test, y_test_pred, average="macro"), 4),
            "test_recall_macro":     round(recall_score(y_test, y_test_pred, average="macro"), 4),
            "test_roc_auc_ovr":      round(roc_auc_score(y_test, y_test_proba, multi_class="ovr"), 4),
            "inference_latency_ms":  infer_latency,
        }
        mlflow.log_metrics(test_metrics)
        logger.info(f"  Test metrics: {test_metrics}")

        os.makedirs("data/evaluation", exist_ok=True)
        class_names = params["data"]["class_names"]

        cm_path = params["evaluation"]["confusion_matrix_path"]
        plot_confusion_matrix(y_test, y_test_pred, class_names, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="evaluation")

        roc_path = params["evaluation"]["roc_curve_path"]
        plot_roc_curve(y_test, y_test_proba, class_names, roc_path)
        mlflow.log_artifact(roc_path, artifact_path="evaluation")
        report_path = params["evaluation"]["classification_report_path"]
        report      = save_classification_report(
            y_test, y_test_pred, class_names, report_path
        )
        mlflow.log_artifact(report_path, artifact_path="evaluation")
        logger.info(f"\n{report}")
        logger.info("Logging model to MLflow...")
        sample_input  = X_train[:5]
        sample_output = pipeline.predict(sample_input)
        signature     = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            sk_model        = pipeline,
            artifact_path   = "model",
            signature       = signature,
            input_example   = sample_input,
            registered_model_name = params["training"]["model_name"],
        )

        logger.info(
            f"Model logged to MLflow | "
            f"run_id={run_id}"
        )
        criteria     = params["training"]["acceptance_criteria"]
        test_f1      = test_metrics["test_f1_macro"]
        test_acc     = test_metrics["test_accuracy"]
        test_latency = test_metrics["inference_latency_ms"]

        criteria_met = (
            test_f1      >= criteria["min_f1_score"] and
            test_acc     >= criteria["min_accuracy"] and
            test_latency <= criteria["max_inference_latency_ms"]
        )

        mlflow.log_metric("acceptance_criteria_met", int(criteria_met))

        if criteria_met and params["training"]["promote_to_staging"]:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(
                params["training"]["model_name"],
                stages=["None"]
            )
            if versions:
                latest_version = versions[-1].version
                client.transition_model_version_stage(
                    name    = params["training"]["model_name"],
                    version = latest_version,
                    stage   = "Staging",
                )
                logger.info(
                    f"Model v{latest_version} promoted to Staging! "
                    f"(F1={test_f1:.4f}, Acc={test_acc:.4f}, "
                    f"Latency={test_latency}ms)"
                )
        else:
            if not criteria_met:
                logger.warning(
                    f"Model did NOT meet acceptance criteria. "
                    f"Not promoted to Staging.\n"
                    f"  F1={test_f1:.4f} (min={criteria['min_f1_score']})\n"
                    f"  Acc={test_acc:.4f} (min={criteria['min_accuracy']})\n"
                    f"  Latency={test_latency}ms "
                    f"(max={criteria['max_inference_latency_ms']}ms)"
                )

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE — SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Run ID:            {run_id}")
        logger.info(f"  Model:             {params['model']['active_model']}")
        logger.info(f"  Test F1 (macro):   {test_f1:.4f}")
        logger.info(f"  Test Accuracy:     {test_acc:.4f}")
        logger.info(f"  Inference Latency: {test_latency}ms")
        logger.info(f"  CV Score:          {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        logger.info(f"  Criteria Met:      {'YES' if criteria_met else 'NO'}")
        logger.info(f"  MLflow UI:         {mlflow_uri}/#/experiments")
        logger.info("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument(
        "--params",
        type=str,
        default="training/params.yaml",
        help="Path to params.yaml file",
    )
    args = parser.parse_args()

    train(params_path=args.params)
