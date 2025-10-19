import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from .constant import (
    PRED_TRUE_DIR,
    PRED_TRUE_CSV,
    CURRENCIES,
    RESULTS_PATH,
)


def load_metrics():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_pred_true(currency):
    csv_path = PRED_TRUE_DIR / f"{currency}_{PRED_TRUE_CSV}"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    return df


def generate_metrics_table(metrics_dict: dict) -> str:
    rows = []
    for tgt, m in metrics_dict.items():
        if m is None:
            rows.append(
                f"<tr><td colspan='4' style='border:1px solid black; text-align:center;'>{tgt.upper()}: 데이터 부족</td></tr>"
            )
        else:
            rows.append(
                f"<tr>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{tgt.upper()}</td>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{m['rmse']:.4f}</td>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{m['r2']:.4f}</td>"
                f"<td style='border:1px solid black; padding:6px; text-align:center;'>{m['mape']:.2f}%</td>"
                f"</tr>"
            )
    return (
        "<div style='display:flex; justify-content:flex-end; margin-top:30px;'>"
        "<table style='border-collapse: collapse; font-size: 16px; margin-bottom: 40px;'>"
        "<thead>"
        "<tr><th>통화</th><th>RMSE</th><th>R²</th><th>MAPE</th></tr>"
        "</thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></div>"
    )


app = FastAPI(title="환율 예측 대시보드")


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    html = """
    <html><head><title>환율 예측 대시보드</title></head>
    <body>
        <h2>환율 예측 대시보드</h2>
        <p><a href='/train'>/train</a> - 전체 예측</p>
        <p><a href='/train_visual'>/train_visual</a> - 성능 시각화</p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/train", response_class=HTMLResponse)
async def train() -> HTMLResponse:
    metrics = load_metrics()
    selected_targets = CURRENCIES
    fig = make_subplots(
        rows=len(selected_targets),
        cols=1,
        subplot_titles=[f"{t.upper()} 환율 예측" for t in selected_targets],
    )
    metrics_dict = {}

    for i, tgt in enumerate(selected_targets, 1):
        df = load_pred_true(tgt)
        m = metrics.get(tgt, {}).get("metrics")
        metrics_dict[tgt] = m
        if df is None or m is None:
            continue
        if "date" in df.columns:
            x = df["date"]
        else:
            x = df.index
        fig.add_trace(
            go.Scatter(x=x, y=df["true"], mode="lines", name=f"{tgt.upper()} 실제값"),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["pred"], mode="lines", name=f"{tgt.upper()} 예측값"),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=600 * len(selected_targets), title="환율 예측 결과", showlegend=True
    )
    return HTMLResponse(
        fig.to_html(full_html=False, include_plotlyjs="cdn")
        + generate_metrics_table(metrics_dict)
    )


@app.get("/train_visual", response_class=HTMLResponse)
async def train_visual() -> HTMLResponse:
    metrics = load_metrics()
    heat_values = {"target": [], "rmse": [], "mape": [], "r2": []}
    residual_traces, scatter_traces = [], []
    y_true_all, y_pred_all = [], []

    for tgt in CURRENCIES:
        m = metrics.get(tgt, {}).get("metrics")
        df = load_pred_true(tgt)
        if m is None or df is None:
            continue
        heat_values["target"].append(tgt.upper())
        heat_values["rmse"].append(m["rmse"])
        heat_values["mape"].append(m["mape"])
        heat_values["r2"].append(m["r2"])

        y_true = df["true"].values
        y_pred = df["pred"].values
        residual_traces.append(go.Box(y=(y_true - y_pred), name=tgt.upper()))
        scatter_traces.append(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name=tgt.upper(),
                opacity=0.5,
            )
        )
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

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
    heatmap_fig.update_layout(title="모델 성능 지표", height=400)

    box_fig = go.Figure(data=residual_traces)
    box_fig.update_layout(title="Residual 분포", height=400)

    scatter_fig = go.Figure(data=scatter_traces)
    if y_true_all and y_pred_all:
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

    return HTMLResponse(
        "<h2>결과 분석 시각화</h2>"
        + heatmap_fig.to_html(full_html=False, include_plotlyjs="cdn")
        + box_fig.to_html(full_html=False, include_plotlyjs=False)
        + scatter_fig.to_html(full_html=False, include_plotlyjs=False)
    )


if __name__ == "__main__":
    import uvicorn

    print(os.getcwd())
    uvicorn.run("dashboard:app", host="127.0.0.1", port=8000, reload=True)
