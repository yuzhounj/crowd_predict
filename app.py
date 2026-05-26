"""FastAPI 预测服务 —— 本地部署。"""
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import date, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="南京景点客流预测")

MODEL_DIR = "models/saved"

# ============================================================
# 启动时加载模型和数据
# ============================================================

models: dict = {}
meta: dict = {}
full_df: pd.DataFrame = None
attractions: list = []
time_points = ["9:30", "11:30", "13:30", "15:30"]


# ============================================================
# 节假日判定
# ============================================================


def is_holiday(d: date) -> bool:
    try:
        from chinese_calendar import is_holiday as ch_is_holiday
        return ch_is_holiday(d)
    except ImportError:
        return False


def is_winter_vacation(d: date) -> bool:
    m, day = d.month, d.day
    if m == 1 and day >= 15:
        return True
    if m == 2 and day <= 15:
        return True
    return False


def is_summer_vacation(d: date) -> bool:
    return d.month in (7, 8)


# 数据中最后一年
MAX_DATE = None


@app.on_event("startup")
def load_all():
    global models, meta, full_df, attractions, MAX_DATE

    with open(f"{MODEL_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    attractions = list(meta["attraction_map"].keys())

    for a in attractions:
        with open(f"{MODEL_DIR}/xgb_{a}.pkl", "rb") as f:
            models[a] = pickle.load(f)

    full_df = pd.read_csv(f"{MODEL_DIR}/full_features.csv", encoding="utf-8-sig")
    full_df["date"] = pd.to_datetime(full_df["date"])
    MAX_DATE = full_df["date"].max().date()
    print(f"已加载 {len(models)} 个模型，{len(full_df)} 条特征数据，数据截止 {MAX_DATE}")


# ============================================================
# 特征计算
# ============================================================


def _find_reference_date(dt: date) -> date:
    """把未来日期映射到数据范围内的同期日期（同月同日，取最后一年）。"""
    if dt <= MAX_DATE:
        return dt
    # 映射到数据最后一年同月同日
    import calendar
    try:
        ref = date(MAX_DATE.year, dt.month, dt.day)
    except ValueError:
        # 2月29日 -> 2月28日
        ref = date(MAX_DATE.year, dt.month, calendar.monthrange(MAX_DATE.year, dt.month)[1])
    return ref


def compute_features(
    attraction: str, dt: date, tp: str, temp: float, precip: float
) -> pd.DataFrame:
    """为预测请求构造特征行。直接复用训练数据中已有的 past_7d_avg / lag 值。"""
    holiday_flag = is_holiday(dt)
    weekend_flag = dt.weekday() >= 5 and not holiday_flag
    vac_flag = (is_winter_vacation(dt) or is_summer_vacation(dt)) and not holiday_flag

    ref_dt = _find_reference_date(dt)
    ref_row = full_df[
        (full_df["attraction"] == attraction)
        & (full_df["time_point"] == tp)
        & (full_df["date"] == pd.Timestamp(ref_dt))
    ]

    if len(ref_row) > 0:
        past_7d = ref_row["past_7d_avg"].values[0]
        lag1 = ref_row["lag_1"].values[0]
        lag2 = ref_row["lag_2"].values[0]
    else:
        # 兜底：取该景区+时点的均值
        fallback = full_df[(full_df["attraction"] == attraction) & (full_df["time_point"] == tp)]["visitors"].mean()
        past_7d = lag1 = lag2 = fallback

    row = {
        "attraction_encoded": meta["attraction_map"][attraction],
        "time_point_encoded": meta["time_map"][tp],
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "temperature": temp,
        "precipitation": precip,
        "is_weekend": int(weekend_flag),
        "is_holiday": int(holiday_flag),
        "is_vacation": int(vac_flag),
        "past_7d_avg": round(past_7d),
        "lag_1": round(lag1),
        "lag_2": round(lag2),
    }
    return pd.DataFrame([row])


# ============================================================
# API 路由
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/api/predict")
async def predict(request: Request):
    body = await request.json()
    attraction = body["attraction"]
    dt = date.fromisoformat(body["date"])
    tp = body["time_point"]
    temp = float(body["temperature"])
    precip = float(body["precipitation"])

    features = compute_features(attraction, dt, tp, temp, precip)
    model = models[attraction]
    pred = int(model.predict(features)[0])

    # 同时算 4 个时点的预测
    all_predictions = {}
    for t in time_points:
        if t == tp:
            all_predictions[t] = pred
        else:
            f2 = compute_features(attraction, dt, t, temp, precip)
            all_predictions[t] = int(model.predict(f2)[0])

    return JSONResponse({
        "attraction": attraction,
        "date": str(dt),
        "predictions": all_predictions,
    })


@app.get("/api/meta")
async def get_meta():
    return JSONResponse({
        "attractions": attractions,
        "time_points": time_points,
    })


# ============================================================
# HTML 前端
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>南京景点客流预测</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
  }
  .container {
    background: white;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    max-width: 600px;
    width: 100%;
    padding: 40px;
  }
  h1 {
    text-align: center;
    color: #333;
    margin-bottom: 8px;
    font-size: 24px;
  }
  .subtitle {
    text-align: center;
    color: #999;
    font-size: 13px;
    margin-bottom: 30px;
  }
  .form-group {
    margin-bottom: 18px;
  }
  label {
    display: block;
    font-size: 14px;
    color: #555;
    margin-bottom: 6px;
    font-weight: 600;
  }
  select, input {
    width: 100%;
    padding: 10px 14px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 15px;
    transition: border-color 0.2s;
    outline: none;
    background: #fafafa;
  }
  select:focus, input:focus {
    border-color: #667eea;
    background: white;
  }
  .row {
    display: flex;
    gap: 12px;
  }
  .row > * { flex: 1; }
  button {
    width: 100%;
    padding: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.1s, box-shadow 0.2s;
    margin-top: 8px;
  }
  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102,126,234,0.4);
  }
  button:active { transform: translateY(0); }
  button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
  .result {
    margin-top: 24px;
    padding: 20px;
    background: #f5f3ff;
    border-radius: 12px;
    display: none;
  }
  .result.show { display: block; }
  .result h3 {
    color: #5b21b6;
    margin-bottom: 16px;
    font-size: 16px;
  }
  .time-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  .time-card {
    text-align: center;
    padding: 14px 8px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .time-card.active {
    border: 2px solid #667eea;
    background: #ede9fe;
  }
  .time-card .time-label {
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
  }
  .time-card .time-value {
    font-size: 20px;
    font-weight: 700;
    color: #333;
  }
  .error {
    background: #fef2f2;
    color: #dc2626;
    padding: 12px;
    border-radius: 8px;
    margin-top: 12px;
    display: none;
  }
</style>
</head>
<body>
<div class="container">
  <h1>南京景点客流预测</h1>
  <p class="subtitle">基于 XGBoost 模型 &middot; 历史客流 + 天气特征</p>

  <form id="predictForm">
    <div class="form-group">
      <label>景区</label>
      <select id="attraction" required></select>
    </div>

    <div class="row">
      <div class="form-group">
        <label>日期</label>
        <input type="date" id="date" required>
      </div>
      <div class="form-group">
        <label>时点</label>
        <select id="time_point" required></select>
      </div>
    </div>

    <div class="row">
      <div class="form-group">
        <label>温度（℃）</label>
        <input type="number" id="temperature" placeholder="例：27" step="0.1" required>
      </div>
      <div class="form-group">
        <label>降水量（mm）</label>
        <input type="number" id="precipitation" placeholder="例：3.0" step="0.1" value="0" required>
      </div>
    </div>

    <button type="submit" id="submitBtn">预测客流</button>
  </form>

  <div class="error" id="error"></div>

  <div class="result" id="result">
    <h3 id="resultTitle"></h3>
    <div class="time-grid" id="timeGrid"></div>
  </div>
</div>

<script>
  // 初始化
  fetch('/api/meta')
    .then(r => r.json())
    .then(data => {
      const selA = document.getElementById('attraction');
      data.attractions.forEach(a => {
        const opt = document.createElement('option');
        opt.value = a; opt.textContent = a;
        selA.appendChild(opt);
      });
      const selT = document.getElementById('time_point');
      data.time_points.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t; opt.textContent = t;
        selT.appendChild(opt);
      });
    });

  // 默认日期：明天
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  document.getElementById('date').value = tomorrow.toISOString().slice(0, 10);

  // 提交
  document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const errorEl = document.getElementById('error');
    const resultEl = document.getElementById('result');
    btn.disabled = true;
    btn.textContent = '预测中...';
    errorEl.style.display = 'none';
    resultEl.classList.remove('show');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          attraction: document.getElementById('attraction').value,
          date: document.getElementById('date').value,
          time_point: document.getElementById('time_point').value,
          temperature: parseFloat(document.getElementById('temperature').value),
          precipitation: parseFloat(document.getElementById('precipitation').value),
        })
      });
      if (!res.ok) throw new Error((await res.json()).detail || '请求失败');
      const data = await res.json();

      const timeGrid = document.getElementById('timeGrid');
      timeGrid.innerHTML = '';
      const selTp = document.getElementById('time_point').value;
      for (const [tp, val] of Object.entries(data.predictions)) {
        const card = document.createElement('div');
        card.className = 'time-card' + (tp === selTp ? ' active' : '');
        card.innerHTML = `<div class="time-label">${tp}</div><div class="time-value">${val.toLocaleString()}</div>`;
        timeGrid.appendChild(card);
      }
      document.getElementById('resultTitle').textContent =
        `${data.attraction} · ${data.date} 客流预测`;
      resultEl.classList.add('show');
    } catch (err) {
      errorEl.textContent = '错误: ' + err.message;
      errorEl.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = '预测客流';
    }
  });
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
