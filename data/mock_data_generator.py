"""Mock 数据生成器 —— 南京热门景点客流 + 天气数据。

数据锚点：2026-05-26（周二）15:30，小雨（~3mm），27℃
- 钟山风景名胜区：19,529
- 南京博物院：8,137
- 红山森林动物园：4,053
- 牛首山文化旅游区：2,266
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

# ============================================================
# 配置
# ============================================================

ATTRACTIONS = ["钟山风景名胜区", "红山森林动物园", "南京博物院", "牛首山文化旅游区"]
TIME_POINTS = ["9:30", "11:30", "13:30", "15:30"]

# 各景区 15:30 晴天工作日基准客流（从锚点反推：19529/0.88≈22200, 8137/0.88≈9250, 4053/0.88≈4600, 2266/0.88≈2580）
BASE_15_30 = {
    "钟山风景名胜区": 22200,
    "南京博物院": 9250,
    "红山森林动物园": 4600,
    "牛首山文化旅游区": 2580,
}

# 时点系数
TIME_RATIOS = {"9:30": 0.55, "11:30": 0.80, "13:30": 0.95, "15:30": 1.0}

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
np.random.seed(42)

# 寒假 1月15日-2月15日，暑假 7月1日-8月31日
WINTER_VACATION = [(1, 15), (2, 15)]
SUMMER_VACATION = [(7, 1), (8, 31)]

# ============================================================
# 节假日判定
# ============================================================


def is_holiday(d: date) -> bool:
    """使用 chinese-calendar 库判定是否为节假日。"""
    try:
        from chinese_calendar import is_holiday as ch_is_holiday
        return ch_is_holiday(d)
    except ImportError:
        return _fallback_holiday(d)


def _fallback_holiday(d: date) -> bool:
    """手动节假日回退逻辑（2024-2025）。"""
    holidays = set()
    # 2024
    holidays.add(date(2024, 1, 1))  # 元旦
    for i in range(8):
        holidays.add(date(2024, 2, 10) + timedelta(days=i))  # 春节 2.10-2.17
    holidays.add(date(2024, 4, 4))  # 清明
    holidays.add(date(2024, 4, 5))
    holidays.add(date(2024, 4, 6))
    for i in range(5):
        holidays.add(date(2024, 5, 1) + timedelta(days=i))  # 五一
    holidays.add(date(2024, 6, 10))  # 端午
    holidays.add(date(2024, 9, 17))  # 中秋
    for i in range(7):
        holidays.add(date(2024, 10, 1) + timedelta(days=i))  # 国庆
    # 2025
    holidays.add(date(2025, 1, 1))
    for i in range(8):
        holidays.add(date(2025, 1, 29) + timedelta(days=i))  # 春节 1.29-2.5
    holidays.add(date(2025, 4, 5))
    for i in range(3):
        holidays.add(date(2025, 4, 5) + timedelta(days=i))
    for i in range(5):
        holidays.add(date(2025, 5, 1) + timedelta(days=i))
    holidays.add(date(2025, 5, 31))  # 端午
    holidays.add(date(2025, 10, 6))  # 中秋
    for i in range(7):
        holidays.add(date(2025, 10, 1) + timedelta(days=i))
    return d in holidays


def is_winter_vacation(d: date) -> bool:
    m, day = d.month, d.day
    if m == 1 and day >= WINTER_VACATION[0][1]:
        return True
    if m == 2 and day <= WINTER_VACATION[1][1]:
        return True
    return False


def is_summer_vacation(d: date) -> bool:
    m, day = d.month, d.day
    if m == 7 or m == 8:
        return True
    if m == 9 and day <= 1:
        return False
    return False


# ============================================================
# 天气生成
# ============================================================


def generate_weather(dates: list[date]) -> pd.DataFrame:
    """为日期序列生成南京天气数据（温度 + 降水量）。"""
    records = []
    for d in dates:
        doy = d.timetuple().tm_yday  # 1-365

        # 温度：年周期 + 日随机
        base_temp = 15 + 12 * np.sin(2 * np.pi * (doy - 110) / 365)  # 峰值在 7 月底
        temp = base_temp + np.random.normal(0, 2.5)
        temp = round(temp, 1)

        # 降水量：夏季多雨，冬季少雨
        rain_base = (np.sin(2 * np.pi * (doy - 100) / 365) + 1) / 2  # 0-1，夏高冬低
        rain_season = rain_base * 6  # 0-6mm 基准
        # 约 30% 天数有雨
        if np.random.random() < 0.30 * (0.5 + rain_base):
            precip = max(0, np.random.exponential(rain_season + 1))
        else:
            precip = 0.0
        precip = round(precip, 1)

        records.append({"date": d, "temperature": temp, "precipitation": precip})

    return pd.DataFrame(records)


# ============================================================
# 客流生成
# ============================================================


def _season_factor(d: date) -> float:
    """季节效应系数。"""
    m = d.month
    if m in (3, 4, 10, 11):  # 春秋旺季
        return np.random.uniform(1.2, 1.5)
    elif m in (12, 1, 2):  # 冬季淡季
        return np.random.uniform(0.70, 0.85)
    else:  # 5-9 月正常
        return 1.0


def _temp_factor(temp: float) -> float:
    """温度效应：15-25℃ 最佳，<5℃ 或 >35℃ 下降 15-25%。"""
    if temp < 5:
        return 1.0 - np.random.uniform(0.15, 0.25)
    elif temp > 35:
        return 1.0 - np.random.uniform(0.15, 0.25)
    elif 15 <= temp <= 25:
        return 1.0
    elif temp < 15:
        return 1.0 - (15 - temp) / 15 * np.random.uniform(0.05, 0.12)
    else:  # 25-35
        return 1.0 - (temp - 25) / 10 * np.random.uniform(0.05, 0.12)


def _rain_factor(precip: float) -> float:
    """降雨效应：每 mm 降 1.5%。"""
    if precip <= 0:
        return 1.0
    elif precip < 5:
        return 1.0 - np.random.uniform(0.05, 0.15)
    elif precip < 15:
        return 1.0 - np.random.uniform(0.15, 0.30)
    else:
        return 1.0 - np.random.uniform(0.30, 0.50)


def _weekend_factor(attraction: str) -> float:
    """周末效应系数，各景区略有差异。"""
    if attraction == "牛首山文化旅游区":
        return np.random.uniform(1.8, 2.2)  # 远郊景区周末加成更大
    elif attraction == "南京博物院":
        return np.random.uniform(1.5, 1.8)
    else:
        return np.random.uniform(1.6, 2.0)


def _holiday_factor() -> float:
    """节假日效应。"""
    return np.random.uniform(2.5, 4.5)


def _vacation_factor(attraction: str) -> float:
    """寒暑假效应。"""
    if attraction == "红山森林动物园":
        return np.random.uniform(1.3, 1.5)  # 亲子游多
    elif attraction == "南京博物院":
        return np.random.uniform(1.05, 1.15)  # 影响较小
    elif attraction == "钟山风景名胜区":
        return np.random.uniform(1.1, 1.3)
    else:  # 牛首山
        return np.random.uniform(1.1, 1.25)


def generate_visitors(dates: list[date], weather_df: pd.DataFrame) -> pd.DataFrame:
    """生成客流数据。"""
    records = []
    for d in dates:
        holiday_flag = is_holiday(d)
        weekend_flag = d.weekday() >= 5 and not holiday_flag
        winter_vac = is_winter_vacation(d) and not holiday_flag
        summer_vac = is_summer_vacation(d) and not holiday_flag
        vac_flag = winter_vac or summer_vac

        is_normal = not (holiday_flag or weekend_flag or vac_flag)

        row = weather_df[weather_df["date"] == d].iloc[0]
        temp = row["temperature"]
        precip = row["precipitation"]

        for attraction in ATTRACTIONS:
            for tp in TIME_POINTS:
                base = BASE_15_30[attraction] * TIME_RATIOS[tp]

                # 叠加效应
                factor = 1.0
                factor *= _season_factor(d)
                factor *= _temp_factor(temp)
                factor *= _rain_factor(precip)

                if weekend_flag:
                    factor *= _weekend_factor(attraction)
                elif holiday_flag:
                    factor *= _holiday_factor()
                elif vac_flag:
                    factor *= _vacation_factor(attraction)

                # 随机噪声
                noise = np.random.uniform(0.88, 1.12)
                visitors = int(base * factor * noise)

                # 确保合理范围
                visitors = max(visitors, int(BASE_15_30[attraction] * TIME_RATIOS[tp] * 0.1))

                records.append({
                    "date": d,
                    "time_point": tp,
                    "attraction": attraction,
                    "visitors": visitors,
                    "temperature": temp,
                    "precipitation": precip,
                    "is_weekend": int(weekend_flag),
                    "is_holiday": int(holiday_flag),
                    "is_vacation": int(vac_flag),
                })

    return pd.DataFrame(records)


# ============================================================
# 计算 past_7d_avg
# ============================================================


def add_past_7d_avg(df: pd.DataFrame) -> pd.DataFrame:
    """为每条记录计算同一时点过去 7 天的平均客流（不含当天）。"""
    df = df.sort_values(["attraction", "time_point", "date"]).reset_index(drop=True)
    past_7d_avg = []

    for _, row in df.iterrows():
        d = row["date"]
        attraction = row["attraction"]
        tp = row["time_point"]
        # 过去 7 天
        mask = (
            (df["attraction"] == attraction)
            & (df["time_point"] == tp)
            & (df["date"] < d)
            & (df["date"] >= d - timedelta(days=7))
        )
        subset = df[mask]
        avg = subset["visitors"].mean() if len(subset) > 0 else row["visitors"]
        past_7d_avg.append(round(avg))

    df["past_7d_avg"] = past_7d_avg
    return df


# ============================================================
# 主入口
# ============================================================


def generate(output_path: str = "data/mock_data.csv") -> pd.DataFrame:
    all_dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]

    print(f"生成天气数据 ({len(all_dates)} 天)...")
    weather_df = generate_weather(all_dates)

    print(f"生成客流数据 ({len(all_dates)} 天 × {len(ATTRACTIONS)} 景区 × {len(TIME_POINTS)} 时点)...")
    df = generate_visitors(all_dates, weather_df)

    print("计算 past_7d_avg...")
    df = add_past_7d_avg(df)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Mock 数据已保存至 {output_path}，共 {len(df)} 条记录")
    return df


if __name__ == "__main__":
    generate()
