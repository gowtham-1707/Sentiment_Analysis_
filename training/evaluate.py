import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_params(params_path: str = "training/params.yaml") -> dict:
    """Load parameters from DVC-tracked params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def load_model_from_registry(
    model_name: str,
    version: str = None,
    stage: str = None,
    mlflow_uri: str = "http://localhost:5000",
):
    
    mlflow.set_tracking_uri(mlflow_uri)

    if version:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model: {model_name} v{version}")
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model: {model_name} ({stage})")
    else:
        raise ValueError("Must specify either --version or --stage.")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from MLflow: {model_uri}")
        return model
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        raise RuntimeError(f"Could not load model: {e}") from e


def load_test_data(params: dict) -> tuple:
    processed_path = params["data"]["processed_path"]
    logger.info(f"Loading test data from {processed_path}...")

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed data not found at: {processed_path}\n"
            f"Run the Airflow data pipeline first."
        )

    data        = joblib.load(processed_path)
    X_test      = data["X_test"]
    y_test      = data["y_test"]
    class_names = params["data"]["class_names"]

    logger.info(f"Test data loaded | samples={len(X_test)}")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        label = class_names[cls]
        pct   = round(cnt / len(y_test) * 100, 1)
        logger.info(f"  {label}: {cnt} samples ({pct}%)")

    return X_test, y_test, class_names

def evaluate_model(
    model,
    X_test: list,
    y_test: np.ndarray,
    class_names: list,
) -> dict:
    
    logger.info("Running inference on test set...")
    infer_start  = time.time()
    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)
    infer_total  = time.time() - infer_start

    inference_latency_ms     = round((infer_total / len(X_test)) * 1000, 4)
    total_inference_time_ms  = round(infer_total * 1000, 2)

    logger.info(
        f"Inference complete | "
        f"total={total_inference_time_ms}ms | "
        f"avg={inference_latency_ms}ms/sample"
    )
    metrics = {
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":              round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_weighted":           round(f1_score(y_test, y_pred, average="weighted"), 4),
        "f1_micro":              round(f1_score(y_test, y_pred, average="micro"), 4),
        "precision_macro":       round(precision_score(y_test, y_pred, average="macro"), 4),
        "precision_weighted":    round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall_macro":          round(recall_score(y_test, y_pred, average="macro"), 4),
        "recall_weighted":       round(recall_score(y_test, y_pred, average="weighted"), 4),
        "roc_auc_ovr":           round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 4),
        "roc_auc_ovo":           round(roc_auc_score(y_test, y_proba, multi_class="ovo"), 4),
        "inference_latency_ms":  inference_latency_ms,
        "total_inference_ms":    total_inference_time_ms,
        "test_samples":          len(X_test),
    }
    per_class_f1        = f1_score(y_test, y_pred, average=None)
    per_class_precision = precision_score(y_test, y_pred, average=None)
    per_class_recall    = recall_score(y_test, y_pred, average=None)

    for i, name in enumerate(class_names):
        key = name.lower()
        metrics[f"f1_{key}"]        = round(float(per_class_f1[i]), 4)
        metrics[f"precision_{key}"] = round(float(per_class_precision[i]), 4)
        metrics[f"recall_{key}"]    = round(float(per_class_recall[i]), 4)

    return metrics, y_pred, y_proba

def generate_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_dir: str,
) -> str:
    """Generate and save confusion matrix heatmap."""
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        thresh = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black",
                    fontsize=11,
                )

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(title, fontsize=12)

    plt.suptitle("Sentiment Classifier — Confusion Matrices", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix_eval.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")
    return path


def generate_roc_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    output_dir: str,
) -> str:
    """Generate and save ROC curves for all classes."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))
    colors    = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax   = plt.subplots(figsize=(8, 6))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves_eval.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved: {path}")
    return path


def generate_metrics_bar_chart(metrics: dict, class_names: list, output_dir: str) -> str:
    categories = ["f1", "precision", "recall"]
    colors     = ["#3498db", "#e74c3c", "#2ecc71"]
    x          = np.arange(len(class_names))
    width      = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (cat, color) in enumerate(zip(categories, colors)):
        values = [metrics[f"{cat}_{name.lower()}"] for name in class_names]
        bars   = ax.bar(x + i * width, values, width, label=cat.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class F1 / Precision / Recall", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "per_class_metrics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class metrics chart saved: {path}")
    return path


def save_full_report(
    metrics: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    model_info: dict,
    output_dir: str,
) -> str:
    report_data = {
        "model_info":   model_info,
        "metrics":      metrics,
        "class_report": classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
        ),
    }
    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    txt_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("SENTIMENT ANALYSIS — EVALUATION REPORT\n")
        f.write("=" * 65 + "\n\n")

        f.write("MODEL INFO\n")
        f.write("-" * 40 + "\n")
        for k, v in model_info.items():
            f.write(f"  {k:<25}: {v}\n")

        f.write("\nOVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        key_metrics = [
            "accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro",
            "roc_auc_ovr", "inference_latency_ms",
        ]
        for k in key_metrics:
            f.write(f"  {k:<30}: {metrics[k]}\n")

        f.write("\nPER-CLASS METRICS\n")
        f.write("-" * 40 + "\n")
        for name in class_names:
            key = name.lower()
            f.write(f"  {name}:\n")
            f.write(f"    F1        : {metrics[f'f1_{key}']}\n")
            f.write(f"    Precision : {metrics[f'precision_{key}']}\n")
            f.write(f"    Recall    : {metrics[f'recall_{key}']}\n")

        f.write("\nDETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))

    logger.info(f"Evaluation reports saved: {json_path}, {txt_path}")
    return txt_path

def check_acceptance_criteria(metrics: dict, params: dict) -> bool:

    criteria = params["training"]["acceptance_criteria"]
    checks   = {
        "F1 Macro":          (metrics["f1_macro"],             ">=", criteria["min_f1_score"]),
        "Accuracy":          (metrics["accuracy"],             ">=", criteria["min_accuracy"]),
        "Inference Latency": (metrics["inference_latency_ms"], "<=", criteria["max_inference_latency_ms"]),
    }

    logger.info("\nACCEPTANCE CRITERIA CHECK")
    logger.info("-" * 50)
    all_passed = True

    for name, (value, op, threshold) in checks.items():
        if op == ">=":
            passed = value >= threshold
        elif op == "<=":
            passed = value <= threshold
        else:
            passed = False

        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name:<22}: {value:.4f} {op} {threshold} → {status}")

        if not passed:
            all_passed = False

    logger.info("-" * 50)
    logger.info(f"  Overall: {'ALL CRITERIA MET' if all_passed else 'CRITERIA NOT MET'}")
    return all_passed

def evaluate(
    version: str = None,
    stage: str = "Production",
    params_path: str = "training/params.yaml",
    log_to_mlflow: bool = True,
) -> dict:

    params    = load_params(params_path)
    mlflow_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        params["mlflow"]["tracking_uri"]
    )
    model_name  = params["training"]["model_name"]
    output_dir  = params["evaluation"]["output_dir"]
    class_names = params["data"]["class_names"]

    os.makedirs(output_dir, exist_ok=True)
    model = load_model_from_registry(
        model_name=model_name,
        version=version,
        stage=stage,
        mlflow_uri=mlflow_uri,
    )
    mlflow.set_tracking_uri(mlflow_uri)
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(
        model_name,
        stages=[stage] if stage else []
    )
    model_version = versions[0].version if versions else version or "unknown"
    run_id        = versions[0].run_id   if versions else "unknown"

    model_info = {
        "model_name":    model_name,
        "model_version": model_version,
        "model_stage":   stage or "versioned",
        "run_id":        run_id,
        "mlflow_uri":    mlflow_uri,
    }
    X_test, y_test, class_names = load_test_data(params)
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, class_names)
    cm_path      = generate_confusion_matrix(y_test, y_pred, class_names, output_dir)
    roc_path     = generate_roc_curves(y_test, y_proba, class_names, output_dir)
    chart_path   = generate_metrics_bar_chart(metrics, class_names, output_dir)
    report_path  = save_full_report(
        metrics, y_test, y_pred, class_names, model_info, output_dir
    )
    criteria_met = check_acceptance_criteria(metrics, params)
    if log_to_mlflow:
        with mlflow.start_run(run_name=f"eval_v{model_version}"):
            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_metric("eval_criteria_met", int(criteria_met))
            mlflow.log_artifact(cm_path,    artifact_path="eval_artifacts")
            mlflow.log_artifact(roc_path,   artifact_path="eval_artifacts")
            mlflow.log_artifact(chart_path, artifact_path="eval_artifacts")
            mlflow.log_artifact(report_path, artifact_path="eval_artifacts")
            mlflow.set_tags({
                **params["mlflow"]["tags"],
                "eval_type":     "standalone_evaluation",
                "model_version": model_version,
            })
        logger.info("Evaluation results logged to MLflow")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Model:             {model_name} v{model_version} ({stage})")
    logger.info(f"  Test Samples:      {metrics['test_samples']}")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted:       {metrics['f1_weighted']:.4f}")
    logger.info(f"  ROC AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
    logger.info(f"  Inference Latency: {metrics['inference_latency_ms']}ms/sample")
    logger.info(f"  Criteria Met:      {'YES' if criteria_met else 'NO'}")
    logger.info(f"  Report saved to:   {output_dir}/")
    logger.info("=" * 60)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a registered MLflow sentiment model on the test set."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific MLflow model version to evaluate (e.g. '3')",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["Production", "Staging", "Archived"],
        help="MLflow model stage to evaluate (default: Production)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="training/params.yaml",
        help="Path to params.yaml",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip logging results back to MLflow",
    )
    args = parser.parse_args()

    evaluate(
        version       = args.version,
        stage         = args.stage,
        params_path   = args.params,
        log_to_mlflow = not args.no_mlflow,
    )