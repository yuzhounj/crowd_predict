"""图表生成：所有图表输出为 PNG。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# 注册中文字体（必须在任何 plot 操作之前）
_FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont(_FONT_PATH)
_CN_PROP = fm.FontProperties(fname=_FONT_PATH)
_FONT_NAME = _CN_PROP.get_name()

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [_FONT_NAME, "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

import seaborn as sns
sns.set_style("whitegrid")
# seaborn 会重置字体，需要再次设置
plt.rcParams["font.sans-serif"] = [_FONT_NAME, "Heiti TC"]
plt.rcParams["font.family"] = "sans-serif"

SAVE_DIR = "output/charts"


def _save(fig, name: str):
    path = f"{SAVE_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [OK] {path}")


# ============================================================
# 图表 1：预测 vs 真实对比图（分景区 × 分时点）
# ============================================================


def plot_pred_vs_actual(
    baseline_result: pd.DataFrame,
    xgb_result: pd.DataFrame,
    prophet_result: pd.DataFrame,
):
    print("\n生成：预测 vs 真实对比图...")
    attractions = baseline_result["attraction"].unique()
    time_points = baseline_result["time_point"].unique()

    # 以 XGBoost 测试集日期为准
    dates = sorted(xgb_result["date"].unique())

    fig, axes = plt.subplots(
        len(attractions), len(time_points),
        figsize=(18, 4 * len(attractions)),
        sharex=True,
    )

    for i, attraction in enumerate(attractions):
        for j, tp in enumerate(time_points):
            ax = axes[i][j] if len(attractions) > 1 else axes[j]

            base_sub = baseline_result[
                (baseline_result["attraction"] == attraction) & (baseline_result["time_point"] == tp)
            ].sort_values("date")
            xgb_sub = xgb_result[
                (xgb_result["attraction"] == attraction) & (xgb_result["time_point"] == tp)
            ].sort_values("date")
            prophet_sub = prophet_result[
                (prophet_result["attraction"] == attraction) & (prophet_result["time_point"] == tp)
            ].sort_values("date")

            ax.plot(base_sub["date"], base_sub["visitors"], "k-", label="Actual", linewidth=1.2)
            ax.plot(base_sub["date"], base_sub["predicted"], "r--", label="Baseline", alpha=0.6, linewidth=0.8)
            ax.plot(xgb_sub["date"], xgb_sub["predicted"], "g--", label="XGBoost", alpha=0.8, linewidth=0.8)
            ax.plot(prophet_sub["date"], prophet_sub["predicted"], "b--", label="Prophet", alpha=0.8, linewidth=0.8)

            ax.set_title(f"{attraction} - {tp}", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    _save(fig, "01_pred_vs_actual")
    plt.close("all")


# ============================================================
# 图表 2：三个模型误差对比柱状图
# ============================================================


def plot_error_comparison(metrics_df: pd.DataFrame):
    print("生成：误差对比柱状图...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = ["RMSE", "MAE", "MAPE(%)"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        vals = [metrics_df[metrics_df["Model"] == m][metric].values[0] for m in metrics_df["Model"].unique()]
        models = metrics_df["Model"].unique()
        ax.bar(models, vals, color=colors)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=10)

    fig.tight_layout()
    _save(fig, "02_error_comparison")
    plt.close("all")


# ============================================================
# 图表 3：XGBoost 特征重要性
# ============================================================


def plot_feature_importance(importance_dict: dict):
    print("生成：特征重要性排名图...")
    attractions = list(importance_dict.keys())
    fig, axes = plt.subplots(1, len(attractions), figsize=(5 * len(attractions), 5))

    for i, attraction in enumerate(attractions):
        ax = axes[i] if len(attractions) > 1 else axes
        imp = importance_dict[attraction]
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        names = [f[0][:12] for f in sorted_imp]
        values = [f[1] for f in sorted_imp]

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(names)))[::-1]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(attraction, fontsize=11)
        ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, "03_feature_importance")
    plt.close("all")


# ============================================================
# 图表 4：Prophet 趋势分解图（在 prophet_model.py 中生成）
# ============================================================


# ============================================================
# 图表 5：残差分布图
# ============================================================


def plot_residuals(
    baseline_result: pd.DataFrame,
    xgb_result: pd.DataFrame,
    prophet_result: pd.DataFrame,
):
    print("生成：残差分布图...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = ["Baseline", "XGBoost", "Prophet"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    for idx, (name, df, color) in enumerate(
        zip(names, [baseline_result, xgb_result, prophet_result], colors)
    ):
        ax = axes[idx]
        residuals = (df["visitors"] - df["predicted"]).values
        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} Residuals", fontsize=12)
        ax.set_xlabel("Error (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        # 标注
        ax.text(
            0.95, 0.95,
            f"Mean: {residuals.mean():.0f}\nStd: {residuals.std():.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.tight_layout()
    _save(fig, "05_residual_distribution")
    plt.close("all")


# ============================================================
# 图表 6：未来预测曲线（用 Prophet 预测未来 30 天）
# ============================================================


def plot_future_forecast(train_df: pd.DataFrame, future_df: pd.DataFrame):
    print("生成：未来预测曲线图...")
    from prophet import Prophet

    attractions = train_df["attraction"].unique()
    fig, axes = plt.subplots(len(attractions), 1, figsize=(14, 4 * len(attractions)))

    for i, attraction in enumerate(attractions):
        ax = axes[i] if len(attractions) > 1 else axes
        sub = train_df[
            (train_df["attraction"] == attraction) & (train_df["time_point"] == "15:30")
        ][["date", "visitors"]].rename(columns={"date": "ds", "visitors": "y"})

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(sub)
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        ax.plot(sub["ds"], sub["y"], "k.", markersize=2, alpha=0.3, label="Historical")
        ax.plot(forecast["ds"], forecast["yhat"], "b-", linewidth=1.5, label="Forecast")
        ax.fill_between(
            forecast["ds"].values,
            forecast["yhat_lower"].values,
            forecast["yhat_upper"].values,
            alpha=0.2,
            color="blue",
            label="90% CI",
        )
        ax.set_title(attraction, fontsize=12)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.axvline(x=sub["ds"].max(), color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    _save(fig, "06_future_forecast")
    plt.close("all")


# ============================================================
# 图表 7：客流热度对比（季节 × 时点 × 景区热力图）
# ============================================================


def plot_heatmap_comparison(df: pd.DataFrame):
    print("生成：客流热度对比图...")
    attractions = df["attraction"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, attraction in enumerate(attractions):
        ax = axes_flat[idx]
        sub = df[df["attraction"] == attraction].copy()
        sub["month"] = sub["date"].dt.month

        # 按月份 + 时点聚合平均客流
        pivot = sub.pivot_table(values="visitors", index="month", columns="time_point", aggfunc="mean")
        # 确保时点顺序
        pivot = pivot[["9:30", "11:30", "13:30", "15:30"]]

        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Avg Visitors"})
        ax.set_title(attraction, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time Point")
        ax.set_ylabel("Month")

    fig.tight_layout()
    _save(fig, "07_heatmap_comparison")
    plt.close("all")
