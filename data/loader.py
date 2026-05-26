"""数据加载与特征工程。"""
import pandas as pd
import numpy as np
from datetime import timedelta


def load_data(path: str = "data/mock_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """为 XGBoost 构造特征表。"""
    df = df.copy()
    df = df.sort_values(["attraction", "time_point", "date"]).reset_index(drop=True)

    # 基本特征
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # 景区和时点编码
    attraction_map = {a: i for i, a in enumerate(df["attraction"].unique())}
    time_map = {t: i for i, t in enumerate(df["time_point"].unique())}
    df["attraction_encoded"] = df["attraction"].map(attraction_map)
    df["time_point_encoded"] = df["time_point"].map(time_map)

    # lag 特征：昨天和前天的同时点客流
    lag_1, lag_2 = [], []
    for _, row in df.iterrows():
        d = row["date"]
        a = row["attraction"]
        tp = row["time_point"]
        mask1 = (df["attraction"] == a) & (df["time_point"] == tp) & (df["date"] == d - timedelta(days=1))
        mask2 = (df["attraction"] == a) & (df["time_point"] == tp) & (df["date"] == d - timedelta(days=2))
        v1 = df[mask1]["visitors"].values
        v2 = df[mask2]["visitors"].values
        lag_1.append(v1[0] if len(v1) > 0 else row["visitors"])
        lag_2.append(v2[0] if len(v2) > 0 else row["visitors"])

    df["lag_1"] = lag_1
    df["lag_2"] = lag_2
    return df


def get_feature_columns() -> list[str]:
    return [
        "attraction_encoded",
        "time_point_encoded",
        "month",
        "day_of_week",
        "temperature",
        "precipitation",
        "is_weekend",
        "is_holiday",
        "is_vacation",
        "past_7d_avg",
        "lag_1",
        "lag_2",
    ]


def split_by_date(df: pd.DataFrame, train_days: int = 700) -> tuple:
    """按时间切分训练集和测试集。"""
    dates = sorted(df["date"].unique())
    train_end = dates[train_days - 1]
    # 测试集取后续 30 天
    test_start = dates[train_days]
    test_end = dates[min(train_days + 29, len(dates) - 1)]

    train = df[df["date"] <= train_end]
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    return train, test
