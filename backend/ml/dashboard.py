import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from fastapi import FastAPI, Query
from plotly.subplots import make_subplots
from fastapi.responses import HTMLResponse
from ml.constant import (PRED_TRUE_DIR, RESULTS_FILE_TEMPLATE, PRED_TRUE_CSV_TEMPLATE, ACCUMULATED_PRED_CSV_TEMPLATE)

base_path = Path(__file__).resolve().parent
ml_path = base_path.parent
root_path = base_path.parent.parent
for p in (root_path, ml_path, base_path):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

CURRENCIES = ["usd", "cny", "jpy", "eur"]
ALL_MODELS = ["attention_lstm", "lstm", "xgboost"]

def load_metrics(model_name: str) -> dict:
    results_path = ml_path / RESULTS_FILE_TEMPLATE.format(model_name=model_name)
    if not results_path.exists():
        return{}
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)

def load_pred_true(currency: str, model_name: str) -> pd.DataFrame | None:
    csv_path = PRED_TRUE_DIR / PRED_TRUE_CSV_TEMPLATE.format(target=currency, model_name=model_name)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df

def generate_metrics_table(metrics_dict: dict) -> str:
    rows = []
    for target, data in metrics_dict.items():
        metrics = data.get("metrics", {})
        rows.append(f"""
                    <tr>
                        <td>{target.upper()}</td>
                        <td>{metrics.get('rmse', 'N/A'):.4f}</td>
                        <td>{metrics.get('r2', 'N/A'):.4f}</td>
                        <td>{metrics.get('mape', 'N/A'):.2f}%</td>
                        <td>{data.get('best_params', {}).get('lstm_units', 'N/A')} / {data.get('best_params', {}).get('dropout_rate', 'N/A')} / {data.get('best_params', {}).get('learning_rate', 'N/A')} / {data.get('best_params', {}).get('batch_size', 'N/A')}</td>
                    </tr>
                """)
    is_attention_lstm = any('lstm_units' in data.get('best_params', {}) for data in metrics_dict.values())
    if not is_attention_lstm:
        header_params = "최적 하이퍼파라미터"
    else:
        header_params = "최적 파라미터 (Units/Dropout/LR/Batch Size)"
    return f"""
    <table class="w-full text-sm text-left text-gray-500">
        <thead class="text-xs text-gray-700 uppercase bg-gray-50">
            <tr>
                <th scope="col" class="px-6 py-3">통화</th>
                <th scope="col" class="px-6 py-3">RMSE</th>
                <th scope="col" class="px-6 py-3">MAPE</th>
                <th scope="col" class="px-6 py-3">R²</th>
                <th scope="col" class="px-6 py-3">{header_params}</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

def generate_performance_plots(data_frames: dict, model_name: str) -> tuple:
    heat_values = {"target": [], "rmse": [], "mape": [], "r2": []}
    residual_traces = []
    scatter_traces = []
    y_true_all, y_pred_all = [], []

    for target, df in data_frames.items():
        if df is None:
            continue

        y_true = df["true"].tolist()
        y_pred = df["pred"].tolist()
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        
        # 잔차(Residual) 계산
        residuals = np.array(y_true) - np.array(y_pred)
        
        # 잔차 분포 (Box Plot)
        residual_traces.append(
            go.Box(
                y=residuals, 
                name=f"{target.upper()} 잔차", 
                boxmean=True
            )
        )

        # 예측 vs 실제 (Scatter Plot)
        scatter_traces.append(
            go.Scatter(
                x=y_true, 
                y=y_pred, 
                mode='markers', 
                name=target.upper(), 
                marker=dict(opacity=0.5)
            )
        )

        # 히트맵 데이터 준비 (성능 지표)
        heat_values["target"].append(target.upper())
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        heat_values["rmse"].append(rmse)
        heat_values["mape"].append(mape)
        heat_values["r2"].append(r2)


    # 히트맵 생성
    z_values = np.array([heat_values["rmse"], heat_values["mape"], heat_values["r2"]])
    text_values = np.round(z_values, 3).astype(str)
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=heat_values["target"],
            y=["RMSE", "MAPE", "R²"],
            colorscale="RdYlGn_r",
            text=text_values,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    heatmap_fig.update_layout(title=f"[{model_name.upper()}] 모델 성능 지표", height=400)

    # 잔차 분포 그래프
    box_fig = go.Figure(data=residual_traces)
    box_fig.update_layout(title="Residual 분포", height=400)

    # 예측 vs 실제 그래프
    scatter_fig = go.Figure(data=scatter_traces)
    if y_true_all and y_pred_all:
        # 이상적인 예측선 (y=x) 추가
        min_val, max_val = min(y_true_all + y_pred_all), max(y_true_all + y_pred_all)
        scatter_fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal",
                line=dict(color="gray", dash="dash"),
            )
        )
    scatter_fig.update_layout(
        title="예측 vs 실제", xaxis_title="실제값", yaxis_title="예측값", height=500
    )

    return heatmap_fig, box_fig, scatter_fig


def generate_time_series_plot(model_name: str, currency: str) -> go.Figure | None:    
    # 학습/검증 기간의 예측-실제값 데이터 로드
    df_train_test = load_pred_true(currency=currency, model_name=model_name)
    if df_train_test is None:
        return None
    
    # 다음 1일 예측값 누적 데이터 로드
    accumulated_path = PRED_TRUE_DIR / ACCUMULATED_PRED_CSV_TEMPLATE.format(model_name=model_name)
    if not accumulated_path.exists():
        df_pred_next = None
    else:
        df_pred_next = pd.read_csv(accumulated_path, index_col=0, parse_dates=True)
        # 해당 통화 컬럼만 선택하고 NaN 제거
        if currency in df_pred_next.columns:
            df_pred_next = df_pred_next[currency].dropna().to_frame()
        else:
            df_pred_next = None


    # 최종 시계열 데이터 결합
    fig = go.Figure()

    # 1. 훈련/검증 실제값
    fig.add_trace(go.Scatter(
        x=df_train_test.index, y=df_train_test["true"], mode='lines', 
        name='실제값 (Actual)', line=dict(color='blue')
    ))

    # 2. 훈련/검증 예측값
    fig.add_trace(go.Scatter(
        x=df_train_test.index, y=df_train_test["pred"], mode='lines', 
        name='예측값 (In-Sample Prediction)', line=dict(color='red', dash='dot')
    ))
    
    # 3. 1일 후 예측값 (누적)
    if df_pred_next is not None and not df_pred_next.empty:
        last_train_date = df_train_test.index[-1]
        last_train_true_value = df_train_test.iloc[-1]['true']
        pred_x = [last_train_date] + df_pred_next.index.tolist()
        pred_y = [last_train_true_value] + df_pred_next[currency].tolist()

        fig.add_trace(go.Scatter(
            x=pred_x, 
            y=pred_y, 
            mode='lines+markers', 
            name='1일 후 예측값 (Next-Day Prediction)', 
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
        
        # 마지막 1일 후 예측값 위에 텍스트 표시
        last_pred_date = df_pred_next.index[-1]
        last_pred_value = df_pred_next[currency].iloc[-1]
        fig.add_annotation(
            x=last_pred_date, 
            y=last_pred_value,
            text=f"{last_pred_value:.4f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
    fig.update_layout(
        title=f"[{model_name.upper()}] {currency.upper()} 환율 시계열 예측",
        xaxis_title="날짜",
        yaxis_title="환율 (KRW)",
        legend_title="데이터 종류",
        hovermode="x unified"
    )

    return fig


def create_dashboard_html(model_name: str, selected_currency: str) -> str:    
    # 1. 데이터 로드
    metrics_dict = load_metrics(model_name)
    data_frames = {c: load_pred_true(c, model_name) for c in CURRENCIES}

    # 2. 그래프 생성
    heatmap_fig, box_fig, scatter_fig = generate_performance_plots(data_frames, model_name)
    time_series_fig = generate_time_series_plot(model_name, selected_currency)
    
    # 3. HTML 구성 요소 생성
    metrics_table_html = generate_metrics_table(metrics_dict)
    heatmap_html = heatmap_fig.to_html(full_html=False, include_plotlyjs="cdn")
    box_html = box_fig.to_html(full_html=False, include_plotlyjs=False)
    scatter_html = scatter_fig.to_html(full_html=False, include_plotlyjs=False)
    time_series_html = time_series_fig.to_html(full_html=False, include_plotlyjs=False) if time_series_fig else "<div>시계열 데이터를 로드할 수 없습니다.</div>"

    # 통화 선택 드롭다운 생성
    currency_options = "".join(
        f'<option value="{c}" {"selected" if c == selected_currency else ""}>{c.upper()}</option>'
        for c in CURRENCIES
    )
    
    # 모델 선택 드롭다운 생성
    model_options = "".join(
        f'<option value="{m}" {"selected" if m == model_name else ""}>{m.upper()}</option>'
        for m in ALL_MODELS
    )

    # 4. 최종 HTML 템플릿 반환
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>환율 예측 대시보드</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .plotly-graph-div {{ margin-top: 1rem; }}
    </style>
</head>
<body class="bg-gray-50 p-6 md:p-10">
    <div class="max-w-7xl mx-auto bg-white p-8 shadow-2xl rounded-xl">
        <h1 class="text-4xl font-extrabold text-gray-900 mb-8">환율 예측 모델 대시보드</h1>
        
        <div class="flex flex-col md:flex-row gap-4 mb-8">
            <div class="w-full md:w-1/2">
                <label for="model-selector" class="block mb-2 text-sm font-medium text-gray-900">모델 선택</label>
                <select id="model-selector" onchange="window.location.href='/?model=' + this.value + '&currency=' + document.getElementById('currency-selector').value"
                        class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 shadow-sm">
                    {model_options}
                </select>
            </div>
            <div class="w-full md:w-1/2">
                <label for="currency-selector" class="block mb-2 text-sm font-medium text-gray-900">시계열 통화 선택</label>
                <select id="currency-selector" onchange="window.location.href='/?model=' + document.getElementById('model-selector').value + '&currency=' + this.value"
                        class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 shadow-sm">
                    {currency_options}
                </select>
            </div>
        </div>

        <!-- 모델 성능 지표 -->
        <div class="mb-10 p-6 bg-white rounded-xl border border-gray-200 shadow-md">
            <h2 class="text-2xl font-bold text-gray-800 mb-4">모델 ({model_name.upper()}) 성능 요약</h2>
            {metrics_table_html}
        </div>

        <!-- 시계열 예측 그래프 (Time Series Plot) -->
        <div class="mb-10 p-6 bg-white rounded-xl border border-gray-200 shadow-md">
            <h2 class="text-2xl font-bold text-gray-800 mb-4">{selected_currency.upper()} 시계열 예측</h2>
            <div id="time-series-plot">
                {time_series_html}
            </div>
        </div>
        
        <!-- 성능 시각화 -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div class="p-6 bg-white rounded-xl border border-gray-200 shadow-md">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">통화별 예측 정확도 (지표 히트맵)</h2>
                <div id="heatmap-plot">
                    {heatmap_html}
                </div>
            </div>

            <div class="p-6 bg-white rounded-xl border border-gray-200 shadow-md">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">예측 vs 실제값 산점도</h2>
                <div id="scatter-plot">
                    {scatter_html}
                </div>
            </div>

            <div class="lg:col-span-2 p-6 bg-white rounded-xl border border-gray-200 shadow-md">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">잔차 분포 (Residuals)</h2>
                <div id="box-plot">
                    {box_html}
                </div>
            </div>
        </div>
        
    </div>
</body>
</html>
    """

# FastAPI 인스턴스
app = FastAPI()

# 루트 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(
    model: str = Query(default="attention_lstm", enum=ALL_MODELS),
    currency: str = Query(default="usd", enum=CURRENCIES),
):
    html_content = create_dashboard_html(model_name=model, selected_currency=currency)
    return HTMLResponse(content=html_content)

from sklearn.metrics import r2_score

if __name__ == "__main__":
    import uvicorn
    print("대시보드 파일입니다.")