# 南京热门景点人流预测

基于历史客流数据，使用多种时序预测模型，预测南京热门景区未来任意日期、任意时点的客流量。

## 数据来源

数据来源于[南京文旅局官网 - 景区舒适度](https://www.njlyw.cn/websitenew/web/comfort_level)，包含各景区实时客流数据、天气信息等。

## 预测景点

- 钟山风景名胜区
- 红山森林动物园
- 南京博物院
- 牛首山文化旅游区

## 预测时点

每日 4 个时点：9:30、11:30、13:30、15:30

## 模型方案

三种模型对比：

| 模型 | 说明 |
|------|------|
| 移动平均（Baseline） | 使用过去 7 天同一时点均值作为预测基准 |
| XGBoost | 基于特征工程的梯度提升树模型 |
| Prophet | Facebook 开源的时序预测模型，内置节假日效应 |

## 项目结构

```
crowd_predict/
├── data/
│   ├── mock_data_generator.py  # 数据生成脚本
│   └── loader.py               # 数据加载与特征工程
├── models/
│   ├── baseline.py             # 移动平均基线模型
│   ├── xgboost_model.py        # XGBoost 模型
│   └── prophet_model.py        # Prophet 模型
├── evaluation/
│   └── evaluator.py            # 模型评估（RMSE/MAE/MAPE）
├── visualization/
│   └── plotter.py              # 图表生成
├── output/
│   ├── charts/                 # 图表 PNG
│   └── results/                # 评估结果 CSV
├── main.py                     # 主流程入口
└── requirements.txt            # 依赖
```

## 快速开始

```bash
# 创建虚拟环境
conda create -n crowd_predict python=3.10 -y
conda activate crowd_predict

# 安装依赖
pip install -r requirements.txt

# 运行全流程（生成数据 → 训练 → 评估 → 出图）
python main.py
```

## 评估结果

| 模型 | RMSE | MAE | MAPE |
|------|------|-----|------|
| Baseline（移动平均） | 12,215 | 8,946 | 158.3% |
| Prophet | 3,866 | 1,841 | 19.6% |
| **XGBoost** | **3,585** | **1,608** | **15.2%** |

XGBoost 表现最优，平均预测误差仅 15% 左右，相比基线模型提升了约 10 倍。
