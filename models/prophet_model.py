"""Prophet 模型：分景区 × 分时点训练。"""
import pandas as pd
import numpy as np
from prophet import Prophet


def _prepare_df(df: pd.DataFrame, attraction: str, time_point: str) -> pd.DataFrame:
    """提取单个景区+时点的数据，整理为 Prophet 格式。"""
    sub = df[(df["attraction"] == attraction) & (df["time_point"] == time_point)].copy()
    sub = sub.rename(columns={"date": "ds", "visitors": "y"})
    return sub[["ds", "y"]]


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """分景区×分时点训练 Prophet（共 16 个小模型）并预测。"""
    attractions = train_df["attraction"].unique()
    time_points = train_df["time_point"].unique()
    all_predictions = []

    for attraction in attractions:
        for tp in time_points:
            train_sub = _prepare_df(train_df, attraction, tp)
            test_sub = test_df[
                (test_df["attraction"] == attraction) & (test_df["time_point"] == tp)
            ].copy()

            if len(train_sub) < 30:
                continue

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
            )
            model.fit(train_sub)

            future = model.make_future_dataframe(periods=len(test_sub) + 30)
            forecast = model.predict(future)

            # 匹配测试集日期
            for _, row in test_sub.iterrows():
                fc = forecast[forecast["ds"] == row["date"]]
                if len(fc) > 0:
                    pred = max(0, int(fc["yhat"].values[0]))
                else:
                    pred = 0

                all_predictions.append({
                    "date": row["date"],
                    "time_point": tp,
                    "attraction": attraction,
                    "visitors": row["visitors"],
                    "predicted": pred,
                })

    return pd.DataFrame(all_predictions)


def get_trend_figures(train_df: pd.DataFrame, save_dir: str) -> dict:
    """用全部训练数据拟合 Prophet 并保存趋势分解图。返回 {attraction: figure}。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)
    fm.fontManager.addfont("/System/Library/Fonts/STHeiti Medium.ttc")
    plt.rcParams["font.sans-serif"] = ["Heiti TC", "STHeiti"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    figures = {}
    attractions = train_df["attraction"].unique()

    for attraction in attractions:
        # 以 15:30 为代表时点
        sub = train_df[
            (train_df["attraction"] == attraction) & (train_df["time_point"] == "15:30")
        ][["date", "visitors"]].copy()
        sub = sub.rename(columns={"date": "ds", "visitors": "y"}).sort_values("ds")

        if len(sub) < 30:
            continue

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(sub)

        fig = model.plot_components(model.predict(model.make_future_dataframe(periods=0)))
        fig.savefig(f"{save_dir}/prophet_trend_{attraction}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        figures[attraction] = f"{save_dir}/prophet_trend_{attraction}.png"

    return figures
