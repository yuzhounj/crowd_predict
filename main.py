"""南京热门景点人流预测 —— 主流程。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mock_data_generator import generate as gen_mock
from data.loader import load_data, build_features, split_by_date
from models.baseline import predict as baseline_predict
from models.xgboost_model import train_and_predict as xgb_train
from models.prophet_model import train_and_predict as prophet_train, get_trend_figures
from evaluation.evaluator import (
    evaluate_all,
    evaluate_by_attraction,
    evaluate_by_time_point,
)
from visualization import plotter


def main():
    os.makedirs("output/charts", exist_ok=True)
    os.makedirs("output/results", exist_ok=True)

    # Step 1: 生成 Mock 数据（如果不存在）
    if not os.path.exists("data/mock_data.csv"):
        print("=" * 60)
        print("Step 1: 生成 Mock 数据...")
        print("=" * 60)
        gen_mock("data/mock_data.csv")
    else:
        print("Mock 数据已存在，跳过生成。")

    # Step 2: 加载数据 & 特征工程
    print("\n" + "=" * 60)
    print("Step 2: 加载数据 & 特征工程...")
    print("=" * 60)
    df = load_data("data/mock_data.csv")
    df = build_features(df)
    train_df, test_df = split_by_date(df, train_days=700)
    print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
    print(f"训练日期: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"测试日期: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")

    # Step 3: 基线模型
    print("\n" + "=" * 60)
    print("Step 3: 基线模型（移动平均）...")
    print("=" * 60)
    baseline_result = baseline_predict(train_df, test_df)

    # Step 4: XGBoost
    print("\n" + "=" * 60)
    print("Step 4: XGBoost 模型...")
    print("=" * 60)
    xgb_result, importance_dict = xgb_train(train_df, test_df)

    # Step 5: Prophet
    print("\n" + "=" * 60)
    print("Step 5: Prophet 模型...")
    print("=" * 60)
    prophet_result = prophet_train(train_df, test_df)

    # Step 6: 评估
    print("\n" + "=" * 60)
    print("Step 6: 模型评估")
    print("=" * 60)
    overall = evaluate_all(baseline_result, xgb_result, prophet_result)
    print("\n=== 整体评估 ===")
    print(overall.to_string(index=False))

    by_att = evaluate_by_attraction(baseline_result, xgb_result, prophet_result)
    print("\n=== 按景区评估 ===")
    print(by_att.to_string(index=False))

    by_tp = evaluate_by_time_point(baseline_result, xgb_result, prophet_result)
    print("\n=== 按时点评估 ===")
    print(by_tp.to_string(index=False))

    # 保存结果
    overall.to_csv("output/results/overall_metrics.csv", index=False)
    by_att.to_csv("output/results/metrics_by_attraction.csv", index=False)
    by_tp.to_csv("output/results/metrics_by_timepoint.csv", index=False)

    # Step 7: 生成图表
    print("\n" + "=" * 60)
    print("Step 7: 生成图表...")
    print("=" * 60)

    plotter.plot_pred_vs_actual(baseline_result, xgb_result, prophet_result)
    plotter.plot_error_comparison(overall)
    plotter.plot_feature_importance(importance_dict)
    plotter.plot_residuals(baseline_result, xgb_result, prophet_result)
    plotter.plot_future_forecast(train_df, None)
    plotter.plot_heatmap_comparison(df)
    get_trend_figures(train_df, "output/charts")

    print("\n" + "=" * 60)
    print("全流程完成！图表保存在 output/charts/，结果保存在 output/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
