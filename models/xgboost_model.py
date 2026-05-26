"""XGBoost 模型：分景区训练。"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from data.loader import get_feature_columns


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """分景区训练 XGBoost 并预测，返回预测结果和特征重要性。"""
    feature_cols = get_feature_columns()
    attractions = train_df["attraction"].unique()
    importance_dict = {}
    all_predictions = []

    for attraction in attractions:
        train_a = train_df[train_df["attraction"] == attraction]
        test_a = test_df[test_df["attraction"] == attraction]

        X_train = train_a[feature_cols]
        y_train = train_a["visitors"]
        X_test = test_a[feature_cols]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # 特征重要性
        importance_dict[attraction] = dict(
            zip(feature_cols, model.feature_importances_)
        )

        result = test_a[["date", "time_point", "attraction", "visitors"]].copy()
        result["predicted"] = preds.astype(int)
        all_predictions.append(result)

    return pd.concat(all_predictions, ignore_index=True), importance_dict
