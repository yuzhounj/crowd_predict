"""训练模型并保存到 models/ 目录，供 web 应用加载。"""
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from xgboost import XGBRegressor
from data.loader import load_data, build_features, get_feature_columns

MODEL_DIR = "models/saved"


def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data("data/mock_data.csv")
    df = build_features(df)
    feature_cols = get_feature_columns()

    attractions = df["attraction"].unique()

    for attraction in attractions:
        sub = df[df["attraction"] == attraction]
        X = sub[feature_cols]
        y = sub["visitors"]

        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            random_state=42, verbosity=0,
        )
        model.fit(X, y)

        filename = f"{MODEL_DIR}/xgb_{attraction}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"已保存: {filename}")

    # 保存编码映射和全部数据（用于预测时查 lag 特征）
    meta = {
        "attraction_map": {a: i for i, a in enumerate(attractions)},
        "time_map": {"9:30": 0, "11:30": 1, "13:30": 2, "15:30": 3},
    }
    with open(f"{MODEL_DIR}/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # 保存完整特征数据，方便查 past_7d_avg / lag 值
    df.to_csv(f"{MODEL_DIR}/full_features.csv", index=False, encoding="utf-8-sig")
    print("已保存 meta 和特征数据")


if __name__ == "__main__":
    train_and_save()
