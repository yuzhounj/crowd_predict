"""基线模型：移动平均。"""
import pandas as pd
import numpy as np


def predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """用过去 7 天同时点均值预测测试集。"""
    predictions = []
    train_sorted = train_df.sort_values(["attraction", "time_point", "date"])

    for _, row in test_df.iterrows():
        a = row["attraction"]
        tp = row["time_point"]
        d = row["date"]

        # 找测试日期前最近 7 条同景区同时点的记录（不一定连续 7 天，取最近 7 条）
        hist = train_sorted[
            (train_sorted["attraction"] == a)
            & (train_sorted["time_point"] == tp)
            & (train_sorted["date"] < d)
        ].tail(7)

        pred = hist["visitors"].mean() if len(hist) > 0 else 0
        predictions.append(max(0, int(pred)))

    result = test_df[["date", "time_point", "attraction", "visitors"]].copy()
    result["predicted"] = predictions
    return result
