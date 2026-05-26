"""模型评估：RMSE / MAE / MAPE。"""
import pandas as pd
import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_all(
    baseline_result: pd.DataFrame,
    xgb_result: pd.DataFrame,
    prophet_result: pd.DataFrame,
) -> pd.DataFrame:
    """对比三个模型的评估指标。"""
    metrics = []
    for name, df in [
        ("Baseline", baseline_result),
        ("XGBoost", xgb_result),
        ("Prophet", prophet_result),
    ]:
        y_true = df["visitors"].values
        y_pred = df["predicted"].values
        metrics.append({
            "Model": name,
            "RMSE": round(rmse(y_true, y_pred), 1),
            "MAE": round(mae(y_true, y_pred), 1),
            "MAPE(%)": round(mape(y_true, y_pred), 2),
        })

    return pd.DataFrame(metrics)


def evaluate_by_attraction(
    baseline_result: pd.DataFrame,
    xgb_result: pd.DataFrame,
    prophet_result: pd.DataFrame,
) -> pd.DataFrame:
    """按景区分别评估。"""
    results = []
    for attraction in baseline_result["attraction"].unique():
        for name, df in [
            ("Baseline", baseline_result),
            ("XGBoost", xgb_result),
            ("Prophet", prophet_result),
        ]:
            sub = df[df["attraction"] == attraction]
            y_true = sub["visitors"].values
            y_pred = sub["predicted"].values
            results.append({
                "Attraction": attraction,
                "Model": name,
                "RMSE": round(rmse(y_true, y_pred), 1),
                "MAE": round(mae(y_true, y_pred), 1),
                "MAPE(%)": round(mape(y_true, y_pred), 2),
            })
    return pd.DataFrame(results)


def evaluate_by_time_point(
    baseline_result: pd.DataFrame,
    xgb_result: pd.DataFrame,
    prophet_result: pd.DataFrame,
) -> pd.DataFrame:
    """按时点分别评估。"""
    results = []
    for tp in baseline_result["time_point"].unique():
        for name, df in [
            ("Baseline", baseline_result),
            ("XGBoost", xgb_result),
            ("Prophet", prophet_result),
        ]:
            sub = df[df["time_point"] == tp]
            y_true = sub["visitors"].values
            y_pred = sub["predicted"].values
            results.append({
                "TimePoint": tp,
                "Model": name,
                "RMSE": round(rmse(y_true, y_pred), 1),
                "MAE": round(mae(y_true, y_pred), 1),
                "MAPE(%)": round(mape(y_true, y_pred), 2),
            })
    return pd.DataFrame(results)
