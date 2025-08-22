import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table

# =========================
# Ruta de datos + cache de hot-reload
# =========================
DATA_PATH = os.getenv(
    "DATA_PATH",
    "STORE_SUMMARY_long_form.xlsx"  # Archivo en el mismo directorio
)
_cache = {"mtime": None, "df": None}

def _clean_week(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # 202533.0 -> 202533
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s

def _week_sort_key(w: str) -> int:
    # '2025-W33' -> 202533 (orden numérico real)
    digits = re.sub(r"\D", "", str(w))
    return int(digits) if digits.isdigit() else 0

def load_data(force: bool=False) -> pd.DataFrame:
    """Lee el Excel si cambió en disco (mtime) o si force=True; sino devuelve el cache."""
    global _cache
    try:
        mtime = os.path.getmtime(DATA_PATH)
    except OSError:
        mtime = None

    if not force and _cache["df"] is not None and _cache["mtime"] == mtime:
        return _cache["df"]

    df = pd.read_excel(DATA_PATH)

    # WEEK normalizado (acepta nombres alternos)
    if "WEEK" in df.columns:
        df["WEEK"] = _clean_week(df["WEEK"])
    else:
        for alt in ["Week", "YR_WK", "Wk"]:
            if alt in df.columns:
                df["WEEK"] = _clean_week(df[alt])
                break
        if "WEEK" not in df.columns:
            raise KeyError("No se encontró la columna WEEK en el archivo.")

    # COUNTRY si falta (por rango de STORE_ID)
    if "COUNTRY" not in df.columns and "STORE_ID" in df.columns:
        df["STORE_ID"] = df["STORE_ID"].astype(int)
        def assign_country(store_id: int) -> str:
            if 2000 <= store_id <= 2099: return "EL SALVADOR"
            if 2100 <= store_id <= 2999: return "GUATEMALA"
            if 3000 <= store_id <= 3999: return "COLOMBIA"
            if 4000 <= store_id <= 4999: return "PERU"
            if store_id >= 5000:         return "OTRO"
            return "UNKNOWN"
        df["COUNTRY"] = df["STORE_ID"].apply(assign_country)

    # KPIs
    df["BIAS_PCT"] = df["BIAS"] * 100 if "BIAS" in df.columns else (df["FCST"] - df["SALES"]) / df["SALES"].replace(0, np.nan) * 100
    df["ACC_PCT"]  = df["ACC"] * 100  if "ACC"  in df.columns else 100 - (np.abs(df["FCST"] - df["SALES"]) / df["SALES"].replace(0, np.nan) * 100)
    df["ABS_ERR"]  = (df["FCST"] - df["SALES"]).abs()

    # Severidad (original con ACC; puedes ajustar si lo necesitas)
    bias_abs = df["BIAS_PCT"].abs()
    acc = df["ACC_PCT"]
    bias_raw = df["BIAS_PCT"]
    conditions = [
        (bias_raw >= 0) & (bias_abs <= 3)  & (acc >= 97),
        (bias_raw >= 0) & (bias_abs <= 7)  & (acc >= 93),
        (bias_raw >= 0) & (bias_abs <= 12) & (acc >= 88),
        (bias_raw >= 0) & (bias_abs <= 20) & (acc >= 80),
    ]
    choices = [1, 2, 3, 4]
    df["SEVERITY"] = np.select(conditions, choices, default=5)

    # Cuadrantes (para la pestaña ACC vs BIAS)
    def quadrant_label(bias, acc_):
        if pd.isna(bias) or pd.isna(acc_): return "No Data"
        if bias < 0:                      return "Critical"
        if acc_ >= 90 and bias <= 5:      return "Excellent"
        if acc_ >= 90 and bias > 5:       return "Good ACC"
        if acc_ < 90 and bias <= 5:       return "Needs Attention"
        return "Critical"
    df["QUADRANT"] = np.vectorize(quadrant_label)(df["BIAS_PCT"], df["ACC_PCT"])
    df["PERFECT_FORECAST"] = df["SALES"]

    if not isinstance(_cache, dict):
        _cache = dict(_cache)
    _cache.update({"mtime": mtime, "df": df})
    return df

# =========================
# Estilos (fondo blanco, paleta verde)
# =========================
UI = {
    "bg_app": "#FFFFFF",
    "bg_panels": "#FFFFFF",
    "bg_plot": "#FFFFFF",
    "bg_paper": "#FFFFFF",
    "grid": "#E6E8ED",
    "axis": "#344054",
    "text": "#101828",
    "muted": "#667085",
    "accent_border": "#D9E7DC"
}
COUNTRY_COLORS = {
    "EL SALVADOR": "#1B5E20",
    "GUATEMALA":   "#43A047",
    "COLOMBIA":    "#2F6F6B",
    "PERU":        "#6B7E5A",
    "OTRO":        "#4E6E5B",
    "UNKNOWN":     "#516E60"
}
QUADRANT_COLORS = {
    "Excellent":        "#1B5E20",
    "Good ACC":         "#43A047",
    "Needs Attention":  "#F59E0B",
    "Critical":         "#C2410C",
    "No Data":          "#667085"
}

def common_layout(title: str, x_title: str, y_title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20, color=UI["text"]), x=0.01, xanchor="left"),
        xaxis=dict(title=dict(text=x_title, font=dict(size=14, color=UI["axis"])),
                   gridcolor=UI["grid"], showgrid=True, zeroline=False, tickfont=dict(color=UI["axis"])),
        yaxis=dict(title=dict(text=y_title, font=dict(size=14, color=UI["axis"])),
                   gridcolor=UI["grid"], showgrid=True, zeroline=False, tickfont=dict(color=UI["axis"])),
        plot_bgcolor=UI["bg_plot"], paper_bgcolor=UI["bg_paper"],
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=12, color=UI["text"]),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.95)", bordercolor=UI["grid"], borderwidth=1,
                    font=dict(size=11, color=UI["text"])),
        margin=dict(l=60, r=160, t=70, b=60)
    )

def _bubble_sizes(series, base=4, scale=0.35, lo=3, hi=14):
    # Burbujas pequeñas para mejor legibilidad
    return (np.abs(series).to_numpy() * scale + base).clip(lo, hi)

# =========================
# Gráficos
# =========================
def create_forecast_vs_sales_chart(filtered_df: pd.DataFrame, size_metric="BIAS_PCT", title_suffix="BIAS"):
    s = _bubble_sizes(filtered_df[size_metric], base=4, scale=0.35, lo=3, hi=14)
    min_val = float(np.nanmin([filtered_df["SALES"].min(), filtered_df["FCST"].min()]))
    max_val = float(np.nanmax([filtered_df["SALES"].max(), filtered_df["FCST"].max()]))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=[min_val, max_val], y=[min_val, max_val], mode="lines",
        line=dict(color="#94A3B8", width=1.2, dash="dash"),
        name="Perfect Forecast", hoverinfo="skip"
    ))
    colors = filtered_df["COUNTRY"].map(lambda c: COUNTRY_COLORS.get(c, "#6C757D"))
    fig.add_trace(go.Scattergl(
        x=filtered_df["SALES"], y=filtered_df["FCST"], mode="markers",
        marker=dict(size=s, color=colors, line=dict(width=0.5, color="#FFFFFF"), opacity=0.9),
        name="Stores",
        customdata=np.stack([filtered_df["STORE_ID"], filtered_df["COUNTRY"], filtered_df["WEEK"],
                             filtered_df["BIAS_PCT"], filtered_df["ACC_PCT"]], axis=-1),
        hovertemplate=(
            "Store: %{customdata[0]}<br>Country: %{customdata[1]}<br>Week: %{customdata[2]}<br>"
            "Sales: %{x:,.0f} units<br>Forecast: %{y:,.0f} units<br>"
            "BIAS: %{customdata[3]:.1f}%<br>ACC: %{customdata[4]:.1f}%<br><extra></extra>"
        ),
        showlegend=False
    ))
    fig.update_layout(common_layout(
        f"Forecast vs Sales (Bubble Size = |{title_suffix}|%)",
        "Actual Sales (units)",
        "Forecast (units)"
    ))
    return fig

def create_acc_bias_scatter(filtered_df: pd.DataFrame):
    fig = go.Figure()
    # Zonas (suaves)
    fig.add_shape(type="rect", x0=-100, y0=90, x1=5,   y1=100, fillcolor="rgba(27,94,32,0.06)", line=dict(width=0))
    fig.add_shape(type="rect", x0=5,    y0=90, x1=100, y1=100, fillcolor="rgba(67,160,71,0.06)", line=dict(width=0))
    fig.add_shape(type="rect", x0=-100, y0=0,  x1=5,   y1=90,  fillcolor="rgba(245,158,11,0.05)", line=dict(width=0))
    fig.add_shape(type="rect", x0=5,    y0=0,  x1=100, y1=90,  fillcolor="rgba(194,65,12,0.05)", line=dict(width=0))
    # Líneas de referencia
    fig.add_hline(y=90, line_dash="dash", line_color="#94A3B8", opacity=0.8)
    fig.add_vline(x=0,  line_dash="dash", line_color="#94A3B8", opacity=0.8)
    fig.add_vline(x=5,  line_dash="dash", line_color="#94A3B8", opacity=0.8)
    fig.add_vline(x=-5, line_dash="dash", line_color="#BAC4D0", opacity=0.5)
    size = _bubble_sizes(filtered_df["BIAS_PCT"], base=6, scale=0.2, lo=5, hi=16)
    colors = filtered_df["QUADRANT"].map(lambda q: QUADRANT_COLORS.get(q, "#6C757D"))
    fig.add_trace(go.Scattergl(
        x=filtered_df["BIAS_PCT"], y=filtered_df["ACC_PCT"], mode="markers",
        marker=dict(size=size, color=colors, line=dict(width=0.6, color="#FFFFFF"), opacity=0.95),
        customdata=np.stack([filtered_df["STORE_ID"], filtered_df["COUNTRY"], filtered_df["WEEK"],
                             filtered_df["SALES"], filtered_df["FCST"], filtered_df["BIAS_PCT"], filtered_df["ACC_PCT"]], axis=-1),
        hovertemplate=(
            "Store: %{customdata[0]}<br>Country: %{customdata[1]}<br>Week: %{customdata[2]}<br>"
            "Sales: %{customdata[3]:,.0f} units<br>Forecast: %{customdata[4]:,.0f} units<br>"
            "BIAS: %{customdata[5]:.1f}%<br>ACC: %{customdata[6]:.1f}%<br><extra></extra>"
        ),
        name="Points"
    ))
    layout = common_layout("Accuracy vs Bias (Quadrant View)", "BIAS (%)", "Accuracy (%)")
    layout.update({
        "xaxis": dict(title=dict(text="BIAS (%)",  font=dict(size=14, color=UI["axis"])),
                      range=[-50, 50], gridcolor=UI["grid"], showgrid=True, zeroline=False),
        "yaxis": dict(title=dict(text="Accuracy (%)", font=dict(size=14, color=UI["axis"])),
                      range=[0, 100], gridcolor=UI["grid"], showgrid=True, zeroline=False)
    })
    fig.update_layout(layout)
    return fig

def create_bias_vs_sales_by_week_chart(filtered_df: pd.DataFrame):
    """
    BIAS (%) vs Ventas Totales (X), coloreado por Semana (ordenada).
    """
    tmp = filtered_df.copy()
    tmp["WEEK_STR"] = tmp["WEEK"].astype(str)
    weeks_sorted = sorted(tmp["WEEK_STR"].unique(), key=_week_sort_key)
    tmp["WEEK_STR"] = pd.Categorical(tmp["WEEK_STR"], categories=weeks_sorted, ordered=True)

    size = _bubble_sizes(tmp["BIAS_PCT"], base=4, scale=0.25, lo=3, hi=12)

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", opacity=0.9)

    # Serie única con color ordinal por semana
    fig.add_trace(go.Scattergl(
        x=tmp["SALES"], y=tmp["BIAS_PCT"],
        mode="markers",
        marker=dict(
            size=size,
            color=tmp["WEEK_STR"].cat.codes,
            colorscale="Greens",
            showscale=False,
            line=dict(width=0.6, color="#FFFFFF"),
            opacity=0.9
        ),
        customdata=np.stack([tmp["STORE_ID"], tmp["COUNTRY"], tmp["WEEK_STR"], tmp["FCST"], tmp["ACC_PCT"]], axis=-1),
        hovertemplate=(
            "Store: %{customdata[0]}<br>Country: %{customdata[1]}<br>Week: %{customdata[2]}<br>"
            "Sales: %{x:,.0f} units<br>Forecast: %{customdata[3]:,.0f} units<br>"
            "ACC: %{customdata[4]:.1f}%<br>BIAS: %{y:.1f}%<br><extra></extra>"
        ),
        name="Points"
    ))

    layout = common_layout("BIAS (%) vs Ventas Totales (color=Semana)", "Ventas Totales (units)", "BIAS (%)")
    layout.update({
        "yaxis": dict(title="BIAS (%)", gridcolor=UI["grid"], showgrid=True, zeroline=False, range=[-50, 50])
    })
    fig.update_layout(layout)
    return fig

def create_bias_histogram(filtered_df: pd.DataFrame, bins=40):
    tmp = filtered_df[np.isfinite(filtered_df["BIAS_PCT"])].copy()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=tmp["BIAS_PCT"], nbinsx=bins, marker=dict(color="#6B7E5A"), opacity=0.85,
        hovertemplate="Count: %{y}<br>Bias: %{x:.1f}%<extra></extra>", name="Bias distribution"
    ))
    # Center at 10% (target)
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8", opacity=0.9)
    fig.add_vline(x=10, line_dash="dot", line_color="#F59E42", opacity=0.95, annotation_text="Target 10%", annotation_position="top right")
    fig.update_layout(common_layout("Distribution of BIAS (%)", "BIAS (%)", "Count"))
    fig.update_xaxes(range=[-50, 50])
    return fig

def create_pareto_error(filtered_df: pd.DataFrame, by="STORE_ID", top_n=20):
    grp = (filtered_df.groupby(by, as_index=False).agg(abs_error=("ABS_ERR", "sum")))
    grp = grp.sort_values("abs_error", ascending=False).head(top_n).reset_index(drop=True)
    total = grp["abs_error"].sum()
    grp["cum_pct"] = (grp["abs_error"].cumsum() / total * 100).round(2) if total > 0 else 0
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=grp[by].astype(str), y=grp["abs_error"], name="Abs Error", marker=dict(color="#2F6F6B"), opacity=0.9), secondary_y=False)
    fig.add_trace(go.Scatter(x=grp[by].astype(str), y=grp["cum_pct"], name="Cumulative %", mode="lines+markers",
                             line=dict(dash="solid", width=2), marker=dict(size=6)), secondary_y=True)
    fig.update_yaxes(title_text="Abs Error (units)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 100], secondary_y=True)
    fig.update_layout(common_layout(f"Pareto of Absolute Forecast Error — Top {top_n} ({by})", by, "Abs Error (units)"))
    return fig

# ======= Heatmap / Matriz unida de BIAS por País–Semana =======
def _week_sort_key_safe(w: str) -> int:
    digits = re.sub(r"\D", "", str(w))
    return int(digits) if digits.isdigit() else 0

def create_bias_matrix_heatmap(filtered_df: pd.DataFrame, aggfunc="mean"):
    """
    Heatmap de BIAS (%) por Semana (filas) x País (columnas).
    Aggrega por 'mean' por defecto.
    """
    tmp = filtered_df.copy()
    tmp["WEEK_STR"] = tmp["WEEK"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)

    pivot = tmp.pivot_table(
        index="WEEK_STR",
        columns="COUNTRY",
        values="BIAS_PCT",
        aggfunc=aggfunc
    )

    pivot = pivot.loc[sorted(pivot.index, key=_week_sort_key_safe)]

    colorscale = [
        [0.0,  "#7F1D1D"],  # rojo oscuro (negativo alto)
        [0.2,  "#DC2626"],
        [0.4,  "#FCA5A5"],
        [0.5,  "#F3F4F6"],  # gris claro cercano a 0
        [0.6,  "#CDE8CF"],
        [0.8,  "#43A047"],  # verde medio
        [1.0,  "#1B5E20"]   # verde oscuro (positivo alto)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(title="Bias %")
    ))

    base_h, row_h = 160, 24
    height = min(1000, base_h + row_h * max(1, pivot.shape[0]))

    layout = common_layout("Matriz de BIAS por País y Semana", "País", "Semana")
    layout.update({"showlegend": False, "height": height})
    fig.update_layout(layout)
    return fig

# =========================
# Tablas (mismo formato)
# =========================
def _table_styles():
    styles = {
        "wrapper": {"background-color":UI["bg_panels"],"border":"1px solid "+UI["grid"],"border-radius":"6px","padding":"10px"},
        "header": {"backgroundColor":"#F3F4F6","color":UI["text"],"fontWeight":"600","border":"1px solid "+UI["grid"]},
        "cell": {"textAlign":"left","font-family":"Segoe UI, Inter, Arial, sans-serif","fontSize":"12px",
                 "padding":"8px","backgroundColor":UI["bg_panels"],"color":UI["text"], "border":"0px"}
    }
    return styles

def create_enhanced_table(filtered_outliers: pd.DataFrame):
    styles = _table_styles()
    if filtered_outliers.empty:
        return html.Div([
            html.H4("Severe Outliers", style={"color": UI["text"], "margin":"0 0 8px 0"}),
            html.Div("No severe outliers for the selected filters.", style={
                "padding":"12px","background-color":UI["bg_panels"],"border":"1px solid "+UI["grid"],
                "border-radius":"6px","color":UI["muted"]
            })
        ])
    t = filtered_outliers.copy()
    t["SALES"]    = t["SALES"].map(lambda x: f"{x:,.0f} units")
    t["FCST"]     = t["FCST"].map(lambda x: f"{x:,.0f} units")
    t["BIAS_PCT"] = t["BIAS_PCT"].map(lambda x: f"{x:.1f}%")
    t["ACC_PCT"]  = t["ACC_PCT"].map(lambda x: f"{x:.1f}%")
    # Ensure all keys are strings for DataTable
    def sanitize(val):
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)
    data = [
        {str(k): sanitize(v) for k, v in row.items()} for row in t.to_dict("records")
    ]
    columns = [
        {"name": str(col), "id": str(col), "type": ("text" if t[col].dtype == object else "numeric")}
        for col in t.columns
    ]
    return html.Div([
        html.H4(f"Severe Outliers ({len(t)} stores)", style={"color": UI["text"], "margin":"0 0 8px 0"}),
        dash_table.DataTable(
            data=data,
            columns=columns,
            style_as_list_view=True,
            style_cell=styles["cell"],
            style_header=styles["header"],
            style_data_conditional=[
                {"if": {"filter_query": "{SEVERITY} = 5"}, "backgroundColor": "#FEE2E2", "color": "#7F1D1D"},
                {"if": {"filter_query": "{SEVERITY} = 4"}, "backgroundColor": "#FEF3C7", "color": "#7C2D12"}
            ],
            page_size=15, sort_action="native", filter_action="native", virtualization=True
        )
    ], style=styles["wrapper"])

def create_bias_country_table(filtered_df: pd.DataFrame):
    """
    Tabla de BIAS promedio por país, alineada y con el mismo formato que la tabla de outliers.
    Responde a los filtros actuales.
    """
    styles = _table_styles()
    if filtered_df.empty or "COUNTRY" not in filtered_df.columns:
        return html.Div([
            html.H4("Average BIAS by Country", style={"color": UI["text"], "margin":"0 0 8px 0"}),
            html.Div("No data available.", style={
                "padding":"12px","background-color":UI["bg_panels"],"border":"1px solid "+UI["grid"],
                "border-radius":"6px","color":UI["muted"]
            })
        ], style=styles["wrapper"])

    g = (filtered_df
         .groupby("COUNTRY", as_index=False)
         .agg(avg_bias_pct=("BIAS_PCT", "mean"),
              stores=("STORE_ID", "nunique"),
              weeks=("WEEK", "nunique"),
              sales_total=("SALES", "sum"))
         .sort_values("avg_bias_pct", ascending=False))

    # Formateo
    g["avg_bias_pct"] = g["avg_bias_pct"].map(lambda x: f"{x:.1f}%")
    g["sales_total"]  = g["sales_total"].map(lambda x: f"{x:,.0f} units")
    def sanitize(val):
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)
    data = [
        {str(k): sanitize(v) for k, v in row.items()} for row in g.to_dict("records")
    ]
    columns = [
        {"name": "Country", "id": "COUNTRY", "type": "text"},
        {"name": "Avg BIAS", "id": "avg_bias_pct", "type": "text"},
        {"name": "Stores", "id": "stores", "type": "numeric"},
        {"name": "Weeks", "id": "weeks", "type": "numeric"},
        {"name": "Total Sales", "id": "sales_total", "type": "text"}
    ]
    return html.Div([
        html.H4("Average BIAS by Country", style={"color": UI["text"], "margin":"0 0 8px 0"}),
        dash_table.DataTable(
            data=data,
            columns=columns,
            style_as_list_view=True,
            style_cell=styles["cell"],
            style_header=styles["header"],
            page_size=10,
            sort_action="native",
            filter_action="native",
            virtualization=True
        )
    ], style=styles["wrapper"])

# =========================
# App (Dash)
# =========================
app = dash.Dash(__name__)
server = app.server  # para despliegues WSGI (gunicorn, etc.)

app.layout = html.Div([
    html.Div([
        html.H1("Forecast Performance Dashboard", style={"color":UI["text"],"margin":"0","fontSize":"24px","fontWeight":"600"}),
        html.P("Executive view of forecast accuracy and bias across countries and weeks",
               style={"color":UI["muted"],"margin":"6px 0 0 0","fontSize":"13px"})
    ], style={"padding":"16px","background-color":UI["bg_paper"],"border-bottom":"2px solid "+UI["accent_border"]}),

    # Auto-refresh cada 30s
    dcc.Interval(id="tick", interval=30*1000, n_intervals=0),

    # Filtros (opciones se actualizan con los datos vivos)
    html.Div([
        html.Div([
            html.Label("Filter by Store", style={"font-weight":"600","color":UI["text"]}),
            dcc.Dropdown(id="store-dropdown", multi=True, placeholder="Select stores (blank = all)", style={"margin-top":"6px", "color":"#000"})
        ], style={"width":"32%","display":"inline-block","verticalAlign":"top","paddingRight":"10px"}),

        html.Div([
            html.Label("Filter by Country", style={"font-weight":"600","color":UI["text"]}),
            dcc.Dropdown(id="country-dropdown", multi=True, placeholder="Select countries (blank = all)", style={"margin-top":"6px", "color":"#000"})
        ], style={"width":"32%","display":"inline-block","verticalAlign":"top","paddingRight":"10px"}),

        html.Div([
            html.Label("Filter by Week", style={"font-weight":"600","color":UI["text"]}),
            dcc.Dropdown(id="week-dropdown", multi=True, placeholder="Select weeks (blank = all)", style={"margin-top":"6px", "color":"#000"})
        ], style={"width":"32%","display":"inline-block","verticalAlign":"top"})
    ], style={"padding":"14px","background-color":UI["bg_panels"],"border-bottom":"1px solid "+UI["grid"]}),

    dcc.Tabs(
        id="tabs", value="tab-bias",
        children=[
            dcc.Tab(label="Forecast vs Sales (BIAS)", value="tab-bias",
                    style={"padding":"8px","background":UI["bg_paper"],"color":UI["axis"]},
                    selected_style={"padding":"8px","background":UI["accent_border"],"color":UI["text"]}),
            dcc.Tab(label="Forecast vs Sales (ACC)", value="tab-acc",
                    style={"padding":"8px","background":UI["bg_paper"],"color":UI["axis"]},
                    selected_style={"padding":"8px","background":UI["accent_border"],"color":UI["text"]}),
            dcc.Tab(label="ACC vs BIAS Analysis", value="tab-acc-bias",
                    style={"padding":"8px","background":UI["bg_paper"],"color":UI["axis"]},
                    selected_style={"padding":"8px","background":UI["accent_border"],"color":UI["text"]}),
            dcc.Tab(label="Severe Outliers", value="tab-table",
                    style={"padding":"8px","background":UI["bg_paper"],"color":UI["axis"]},
                    selected_style={"padding":"8px","background":UI["accent_border"],"color":UI["text"]}),
        ],
        style={"margin":"0","borderTop":"1px solid "+UI["grid"],"borderBottom":"1px solid "+UI["grid"],"background":UI["bg_paper"]}
    ),

    html.Div(id="tab-content", style={"padding":"14px","background":UI["bg_app"]})
], style={"font-family":"Segoe UI, Inter, Arial, sans-serif","background":UI["bg_app"],"minHeight":"100vh"})

# ========= Opciones dinámicas (auto-refresh) =========
@app.callback(
    Output("store-dropdown", "options"),
    Output("country-dropdown", "options"),
    Output("week-dropdown", "options"),
    Input("tick", "n_intervals")
)
def refresh_options(_):
    df = load_data()
    stores    = [{"label": str(s), "value": s} for s in sorted(df["STORE_ID"].unique())]
    countries = [{"label": c, "value": c} for c in sorted(df["COUNTRY"].unique())]
    weeks     = [{"label": w, "value": w} for w in sorted(df["WEEK"].astype(str).unique(), key=_week_sort_key)]
    return stores, countries, weeks

# ========= Contenido principal (auto-refresh) =========
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("store-dropdown", "value"),
    Input("country-dropdown", "value"),
    Input("week-dropdown", "value"),
    Input("tick", "n_intervals")
)
def render_content(tab, selected_stores, selected_countries, selected_weeks, _):
    df = load_data()  # recarga si cambió el archivo

    f = df
    if selected_stores:
        f = f[f["STORE_ID"].isin(selected_stores)]
    if selected_countries:
        f = f[f["COUNTRY"].isin(selected_countries)]
    if selected_weeks:
        sel_weeks = set(map(str, selected_weeks))
        f = f[f["WEEK"].astype(str).isin(sel_weeks)]

    if f.empty:
        return html.Div("No data available for the selected filters.",
                        style={"padding":"20px","textAlign":"center","color":UI["muted"]})

    if tab == "tab-bias":
        fig_main        = create_forecast_vs_sales_chart(f, "BIAS_PCT", "BIAS")
        fig_bias_sales  = create_bias_vs_sales_by_week_chart(f)   # X = Ventas totales, color=Semana
        fig_hist        = create_bias_histogram(f)
        fig_pareto      = create_pareto_error(f, by="STORE_ID", top_n=20)
        fig_matrix      = create_bias_matrix_heatmap(f, aggfunc="mean")
        tbl_country     = create_bias_country_table(f)             # NUEVA TABLA

        # 2x2 + heatmap ancho completo + tabla alineada
        return html.Div([
            html.Div([
                html.Div(dcc.Graph(figure=fig_main,        style={"height":"46vh"}), style={"width":"49%","display":"inline-block","verticalAlign":"top"}),
                html.Div(dcc.Graph(figure=fig_bias_sales,  style={"height":"46vh"}), style={"width":"49%","display":"inline-block","marginLeft":"2%","verticalAlign":"top"})
            ], style={"marginBottom":"16px"}),

            html.Div([
                html.Div(dcc.Graph(figure=fig_hist,        style={"height":"42vh"}), style={"width":"49%","display":"inline-block","verticalAlign":"top"}),
                html.Div(dcc.Graph(figure=fig_pareto,      style={"height":"42vh"}), style={"width":"49%","display":"inline-block","marginLeft":"2%","verticalAlign":"top"})
            ], style={"marginBottom":"16px"}),

            html.Div([
                dcc.Graph(figure=fig_matrix)
            ], style={"marginBottom":"16px"}),

            html.Div([
                tbl_country
            ])
        ])

    elif tab == "tab-acc":
        fig = create_forecast_vs_sales_chart(f, "ACC_PCT", "ACC")
        return dcc.Graph(figure=fig, style={"height":"72vh"})

    elif tab == "tab-acc-bias":
        fig = create_acc_bias_scatter(f)
        return dcc.Graph(figure=fig, style={"height":"72vh"})

    elif tab == "tab-table":
        t = f.loc[f["SEVERITY"] >= 4, ["STORE_ID","COUNTRY","WEEK","SALES","FCST","BIAS_PCT","ACC_PCT","SEVERITY"]].copy()
        return create_enhanced_table(t)

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
