"""
CapitalEdge — Dash Dashboard
Premium black-theme real estate intelligence for the DMV metro area.
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(BASE, "data", "processed", "master_dataset.csv")
UV_PATH    = os.path.join(BASE, "data", "processed", "undervalued_zips.csv")
MODEL_PATH = os.path.join(BASE, "models", "best_model.pkl")
MORT_PATH  = os.path.join(BASE, "data", "processed", "mortgage_rates.csv")

FEATURES = [
    "median_household_income", "total_population", "unemployment_rate",
    "college_educated_count", "crimes_per_1000", "median_year_built", "median_gross_rent",
]

# ── palette ───────────────────────────────────────────────────────────────────
C = dict(
    bg      = "#000000",
    surface = "#070707",
    card    = "#0e0e0e",
    card2   = "#141414",
    border  = "#1c1c1c",
    border2 = "#282828",
    text    = "#efefef",
    muted   = "#606060",
    dim     = "#2a2a2a",
    blue    = "#3b82f6",
    blue2   = "#1d4ed8",
    cyan    = "#06b6d4",
    green   = "#22c55e",
    red     = "#ef4444",
    amber   = "#f59e0b",
    purple  = "#a78bfa",
)

CHART = dict(
    paper_bgcolor = "#000000",
    plot_bgcolor  = "#070707",
    font          = dict(color="#efefef", family="Inter,-apple-system,sans-serif", size=12),
    colorway      = ["#3b82f6", "#06b6d4", "#a78bfa", "#f59e0b", "#22c55e", "#ef4444"],
    xaxis         = dict(gridcolor="#131313", zerolinecolor="#131313", linecolor="#1c1c1c",
                         tickfont=dict(color="#606060")),
    yaxis         = dict(gridcolor="#131313", zerolinecolor="#131313", linecolor="#1c1c1c",
                         tickfont=dict(color="#606060")),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                         font=dict(color="#606060")),
    hoverlabel    = dict(bgcolor="#111111", bordercolor="#282828",
                         font=dict(color="#efefef", family="Inter,sans-serif")),
    margin        = dict(l=8, r=8, t=40, b=8),
    title         = dict(font=dict(color="#606060", size=10), x=0, xanchor="left",
                         pad=dict(l=0, b=8)),
)

# ── helpers ───────────────────────────────────────────────────────────────────
def rgba(h, a):
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def monthly_payment(price, rate_pct, down=0.20, years=30):
    if not price or not rate_pct:
        return None
    loan = price * (1 - down)
    r = rate_pct / 100 / 12
    n = years * 12
    return loan * r * (1+r)**n / ((1+r)**n - 1) if r > 0 else loan / n

def usd(v):   return f"${v:,.0f}" if pd.notna(v) and v == v else "—"
def pct(v):   return f"{v*100:+.1f}%" if pd.notna(v) and v == v else "—"
def num(v):   return f"{v:,.0f}" if pd.notna(v) and v == v else "—"
def rate(v):  return f"{v:.2f}%" if pd.notna(v) and v == v else "—"
def xpct(v):  return f"{v:.1f}%" if pd.notna(v) and v == v else "—"

# ── data loading ──────────────────────────────────────────────────────────────
def _load_main():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    latest = df[df["date"] == df["date"].max()].copy()
    return df, latest

def _load_mortgage():
    if os.path.exists(MORT_PATH):
        try:
            return pd.read_csv(MORT_PATH, parse_dates=["date"])
        except Exception:
            pass
    try:
        from io import StringIO
        import requests
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US", timeout=30)
        df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"], na_values=["."])
        df = df.rename(columns={"observation_date": "date", "MORTGAGE30US": "rate"})
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        df.to_csv(MORT_PATH, index=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "rate"])

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def _momentum(df):
    if df.empty:
        return {}
    dates = sorted(df["date"].unique())
    ld = dates[-1]
    p0 = df[df["date"] == ld]["home_value"].median()
    out = {}
    for label, months in [("1M",1),("3M",3),("6M",6),("12M",12)]:
        tgt = ld - pd.DateOffset(months=months)
        closest = min(dates, key=lambda d: abs((d - tgt).days))
        pp = df[df["date"] == closest]["home_value"].median()
        out[label] = (p0 - pp) / pp * 100 if pp else None
    return out

DF, LATEST  = _load_main()
MORT        = _load_mortgage()
MOMENTUM    = _momentum(DF)
HAS_CENSUS  = (not LATEST.empty) and LATEST["median_household_income"].notna().any()
CURR_RATE   = float(MORT["rate"].iloc[-1]) if not MORT.empty else None

# ── UI components ─────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, up=True, accent=None):
    color = accent or C["blue"]
    kids = [
        html.Div(label, style={"color": C["muted"], "fontSize": ".62rem", "fontWeight": "700",
                                "textTransform": "uppercase", "letterSpacing": ".1em"}),
        html.Div(value, style={"color": C["text"], "fontSize": "1.55rem", "fontWeight": "800",
                                "marginTop": "7px", "letterSpacing": "-.025em",
                                "fontVariantNumeric": "tabular-nums"}),
    ]
    if delta:
        kids.append(html.Div(delta, style={
            "color": C["green"] if up else C["red"],
            "fontSize": ".75rem", "marginTop": "5px", "fontWeight": "500",
        }))
    return html.Div(kids, style={
        "background": C["card"], "borderRadius": "10px",
        "border": f"1px solid {C['border']}",
        "borderTop": f"2px solid {color}",
        "padding": "18px 20px", "flex": "1", "minWidth": "0",
    })

def card(title, content, mb="14px", pad="20px"):
    return html.Div([
        html.Div(title, style={
            "color": C["muted"], "fontWeight": "700", "fontSize": ".62rem",
            "textTransform": "uppercase", "letterSpacing": ".1em",
            "marginBottom": "14px", "paddingBottom": "12px",
            "borderBottom": f"1px solid {C['border']}",
        }),
        content,
    ], style={
        "background": C["card"], "borderRadius": "10px",
        "border": f"1px solid {C['border']}",
        "padding": pad, "marginBottom": mb,
    })

def hdr(title, sub=None):
    return html.Div([
        html.H1(title, style={"color": C["text"], "fontWeight": "900", "fontSize": "1.85rem",
                               "margin": "0 0 6px", "letterSpacing": "-.04em"}),
        html.P(sub, style={"color": C["muted"], "margin": "0 0 28px", "fontSize": ".85rem"}) if sub
        else html.Div(style={"marginBottom": "28px"}),
    ])

def g(fig, cfg=None):
    return dcc.Graph(figure=fig, config=cfg or {"displayModeBar": False},
                     style={"background": "transparent"})

def notice(text, icon="ℹ", color=None):
    c = color or C["blue"]
    return html.Div(
        [html.Span(icon + "  ", style={"fontSize": "1rem"}), text],
        style={
            "color": C["muted"], "background": C["card"],
            "border": f"1px solid {C['border']}",
            "borderLeft": f"3px solid {c}",
            "borderRadius": "7px", "padding": "14px 18px",
            "fontSize": ".85rem", "lineHeight": "1.7",
        }
    )

def row(*cols, gap="14px", mb="14px"):
    return html.Div(list(cols),
                    style={"display": "flex", "gap": gap, "marginBottom": mb,
                           "alignItems": "stretch"})

def col(content, flex="1"):
    return html.Div(content, style={"flex": flex, "minWidth": "0"})

# ── app ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width,initial-scale=1"}],
)
server = app.server
app.title = "CapitalEdge"

# ── navigation ────────────────────────────────────────────────────────────────
NAV = [
    ("overview",  "Market Overview"),
    ("pulse",     "Market Pulse"),
    ("compare",   "Neighborhood Compare"),
    ("predictor", "Price Predictor"),
    ("gems",      "Hidden Gems"),
    ("explorer",  "Data Explorer"),
]

def nav_btn(pid, label, active=False):
    return html.Button(
        [html.Span(style={
            "width": "5px", "height": "5px", "borderRadius": "50%",
            "background": C["blue"] if active else C["dim"],
            "marginRight": "11px", "flexShrink": "0", "display": "inline-block",
        }), label],
        id=f"nav-{pid}", n_clicks=0,
        style={
            "display": "flex", "alignItems": "center", "width": "100%",
            "padding": "9px 13px",
            "background": rgba(C["blue"], 0.12) if active else "transparent",
            "border": "none", "borderRadius": "7px",
            "color": C["text"] if active else C["muted"],
            "fontSize": ".83rem", "fontWeight": "600" if active else "400",
            "cursor": "pointer", "textAlign": "left",
            "fontFamily": "Inter,sans-serif", "letterSpacing": ".01em",
        }
    )

# ── layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="page", data="overview"),
    html.Div([
        html.Div([
            html.Span("Capital", style={"color": C["text"], "fontWeight": "900",
                                         "fontSize": "1.18rem", "letterSpacing": "-.03em"}),
            html.Span("Edge", style={"color": C["blue"], "fontWeight": "900",
                                      "fontSize": "1.18rem", "letterSpacing": "-.03em"}),
        ], style={"padding": "26px 18px 4px"}),
        html.Div("DMV REAL ESTATE INTELLIGENCE", style={
            "color": C["dim"], "fontSize": ".56rem", "letterSpacing": ".15em",
            "padding": "0 18px 20px", "fontWeight": "700",
        }),
        html.Div(style={"height":"1px","background":C["border"],"margin":"0 14px 14px"}),
        html.Div(id="nav-buttons",
                 children=[nav_btn(p,l) for p,l in NAV],
                 style={"padding":"0 6px","display":"flex","flexDirection":"column","gap":"1px"}),
        html.Div(style={"flex":"1"}),
        html.Div(style={"height":"1px","background":C["border"],"margin":"0 14px 12px"}),
        html.Div([
            html.Div("Sources", style={"color":C["dim"],"fontSize":".58rem",
                                        "fontWeight":"700","letterSpacing":".1em","marginBottom":"6px"}),
            *[html.Div(s, style={"color":C["dim"],"fontSize":".72rem","lineHeight":"1.9"})
              for s in ["Zillow ZHVI (2000–2026)", "FRED Mortgage Rates", "US Census ACS", "DC Open Data"]],
        ], style={"padding":"0 18px 26px"}),
    ], style={
        "width":"210px","minWidth":"210px","height":"100vh","position":"fixed",
        "top":"0","left":"0","background":"#040404",
        "borderRight":f"1px solid {C['border']}",
        "display":"flex","flexDirection":"column","zIndex":"99","overflowY":"auto",
    }),
    html.Div(id="main", style={
        "marginLeft":"210px","minHeight":"100vh","background":C["bg"],
    }),
], style={
    "fontFamily":"Inter,-apple-system,BlinkMacSystemFont,sans-serif",
    "background":C["bg"],"minHeight":"100vh","WebkitFontSmoothing":"antialiased",
})

# ── navigation callbacks ──────────────────────────────────────────────────────
@app.callback(
    Output("page","data"),
    [Input(f"nav-{p}","n_clicks") for p,_ in NAV],
    prevent_initial_call=True,
)
def set_page(*_):
    ctx = callback_context
    if not ctx.triggered:
        return "overview"
    return ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-","")

@app.callback(Output("nav-buttons","children"), Input("page","data"))
def update_nav(active):
    return [nav_btn(p, l, p==active) for p,l in NAV]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def render_overview():
    if DF.empty:
        return html.Div("No data. Run the pipeline first.",
                        style={"color":C["red"],"padding":"40px"})
    df, lat = DF, LATEST

    med       = lat["home_value"].median()
    yoy_med   = lat["home_value_yoy_change"].median()
    n_zips    = lat["zip_code"].nunique()
    n_counties = lat["county"].nunique()
    rent_med  = lat["rental_value"].median()

    # ── Hero: 26-year state area chart ──
    monthly = df.groupby(["date","state"])["home_value"].median().reset_index()
    fig_hero = go.Figure()
    STATE_CFG = [("DC", C["red"]), ("VA", C["blue"]), ("MD", C["cyan"])]
    for state, color in STATE_CFG:
        sd = monthly[monthly["state"]==state].sort_values("date")
        fig_hero.add_trace(go.Scatter(
            x=sd["date"], y=sd["home_value"], name=state, mode="lines",
            line=dict(color=color, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor=rgba(color, 0.07),
            hovertemplate=f"<b>{state}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    for dt, label in [("2008-09-01","2008 Crisis"),("2020-03-01","COVID-19"),
                       ("2022-03-01","Rate Hikes Begin")]:
        fig_hero.add_vline(x=dt, line_color=C["border2"], line_width=1.5, line_dash="dot")
        fig_hero.add_annotation(x=dt, y=0.97, yref="paper", text=label,
                                 showarrow=False, textangle=-90,
                                 font=dict(color=C["muted"], size=9), xanchor="right")
    fig_hero.update_layout(**CHART, height=340)
    fig_hero.update_yaxes(tickprefix="$", tickformat=",.0f")

    # ── County bubble scatter ──
    cs = lat.groupby(["county","state"]).agg(
        price=("home_value","median"),
        yoy=("home_value_yoy_change","median"),
        n=("zip_code","nunique"),
    ).reset_index()
    cs["yoy_pct"] = cs["yoy"] * 100
    fig_bubble = px.scatter(
        cs, x="yoy_pct", y="price", size="n", color="state",
        text="county",
        color_discrete_map={"DC":C["red"],"VA":C["blue"],"MD":C["cyan"]},
        size_max=28,
        labels={"yoy_pct":"YoY Change (%)","price":"Median Value ($)","n":"# ZIPs"},
        hover_data={"county":True,"n":True,"price":":$,.0f","yoy_pct":":.1f"},
    )
    fig_bubble.update_traces(
        textposition="top center",
        textfont=dict(color=rgba(C["text"],0.5), size=8),
        marker=dict(line=dict(width=1, color=rgba("#fff",0.1))),
    )
    fig_bubble.update_layout(**CHART, height=340, showlegend=True)
    fig_bubble.update_xaxes(ticksuffix="%")
    fig_bubble.update_yaxes(tickprefix="$", tickformat=",.0f")

    # ── Violin: price distribution by state ──
    fig_violin = go.Figure()
    for state, color in STATE_CFG:
        data = lat[lat["state"]==state]["home_value"].dropna()
        fig_violin.add_trace(go.Violin(
            y=data, name=state, box_visible=True, meanline_visible=True,
            fillcolor=rgba(color, 0.18), line_color=color,
            points=False,
            hovertemplate=f"<b>{state}</b><br>$%{{y:,.0f}}<extra></extra>",
        ))
    fig_violin.update_layout(**CHART, height=340, violinmode="overlay")
    fig_violin.update_yaxes(tickprefix="$", tickformat=",.0f")

    # ── Calendar heatmap: YoY by month × year ──
    mm = df.groupby("date")["home_value"].median().reset_index()
    mm["year"]  = mm["date"].dt.year
    mm["month"] = mm["date"].dt.month
    pivot = mm.pivot_table(index="year", columns="month", values="home_value")
    pct_cal = (pivot.pct_change(axis=0) * 100).round(1)
    pct_cal = pct_cal.dropna(how="all").iloc[1:]  # drop first (NaN) year
    pct_cal.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pct_cal.columns)]
    fig_cal = px.imshow(
        pct_cal,
        color_continuous_scale=[[0,"#5c0e0e"],[0.35,"#3a0a0a"],[0.5,"#111111"],
                                  [0.65,"#0a2e16"],[1,"#0d4a23"]],
        color_continuous_midpoint=0,
        aspect="auto",
        labels=dict(x="Month", y="Year", color="YoY %"),
        text_auto=".1f",
    )
    fig_cal.update_layout(**CHART, height=340)
    fig_cal.update_traces(textfont=dict(size=9, color=rgba(C["text"], 0.55)))
    fig_cal.update_coloraxes(colorbar=dict(
        tickfont=dict(color=C["muted"],size=9), ticksuffix="%",
        title=dict(text="YoY %",font=dict(color=C["muted"],size=9)),
        bgcolor="rgba(0,0,0,0)", outlinecolor="rgba(0,0,0,0)",
    ))

    # ── Top appreciating ZIPs ──
    top = (lat.dropna(subset=["home_value_yoy_change"])
              .nlargest(20,"home_value_yoy_change").copy())
    top["yoy_pct"] = (top["home_value_yoy_change"]*100).round(1)
    top["label"] = top["zip_code"].astype(str) + "  " + top["city"].fillna("")
    fig_top = px.bar(
        top.sort_values("yoy_pct"),
        x="yoy_pct", y="label", orientation="h",
        color="yoy_pct",
        color_continuous_scale=[[0,"#083520"],[0.5,"#0d5533"],[1,C["green"]]],
        text="yoy_pct",
        labels={"yoy_pct":"YoY %","label":""},
    )
    fig_top.update_traces(texttemplate="%{text:+.1f}%", textposition="outside",
                           textfont=dict(color=C["muted"],size=10), marker_line_width=0)
    fig_top.update_layout(**CHART, height=520)
    fig_top.update_coloraxes(showscale=False)
    fig_top.update_xaxes(ticksuffix="%")

    # ── Map ──
    map_df = lat.dropna(subset=["latitude","longitude","home_value"])
    if not map_df.empty:
        fig_map = px.scatter_mapbox(
            map_df, lat="latitude", lon="longitude",
            color="home_value", size="home_value", size_max=16,
            hover_name="city",
            hover_data={"zip_code":True,"county":True,"home_value":":$,.0f",
                        "home_value_yoy_change":":.2%","rental_value":":$,.0f",
                        "latitude":False,"longitude":False},
            color_continuous_scale=[[0,"#060e1e"],[0.4,C["blue2"]],[1,C["blue"]]],
            mapbox_style="carto-darkmatter",
            zoom=8, center={"lat":38.9,"lon":-77.04}, height=540,
        )
        fig_map.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
            margin=dict(l=0,r=0,t=0,b=0),
            coloraxis=dict(colorbar=dict(
                tickfont=dict(color=C["muted"],size=9),
                title=dict(text="Value",font=dict(color=C["muted"],size=9)),
                bgcolor="rgba(0,0,0,0)", outlinecolor="rgba(0,0,0,0)",
            ))
        )
        map_el = card("Geographic Price Heat Map — DMV Metro (197 ZIP Codes)", g(fig_map), pad="0px")
    else:
        map_el = notice("Map unavailable. Re-run data_ingestion.py to download ZIP coordinates.", "📍")

    return html.Div([
        hdr("Market Overview","Washington DC Metro Area · Zillow ZHVI 2000–2026"),
        row(kpi("Median Home Value", usd(med)),
            kpi("Median YoY Change", pct(yoy_med), up=(yoy_med or 0)>=0),
            kpi("ZIP Codes Tracked", num(n_zips)),
            kpi("Counties", str(n_counties)),
            kpi("Median Monthly Rent", usd(rent_med), accent=C["purple"]),
            gap="10px", mb="20px"),
        card("26-Year Home Value History · DC vs Virginia vs Maryland", g(fig_hero)),
        row(col(card("County Bubble · YoY% vs Median Value (size = # ZIPs)", g(fig_bubble)), "2"),
            col(card("Home Value Distribution by State", g(fig_violin)), "2")),
        card("YoY Appreciation Heatmap · Every Month Since 2001", g(fig_cal)),
        row(col(card("Top 20 Fastest-Appreciating ZIP Codes", g(fig_top)), "2"),
            col(html.Div([
                card("Quick Stats", html.Div([
                    *[_stat_row(label, val) for label, val in [
                        ("Peak Value",        usd(lat["home_value"].max())),
                        ("Min Value",         usd(lat["home_value"].min())),
                        ("Std Deviation",     usd(lat["home_value"].std())),
                        ("Best County",       lat.groupby("county")["home_value"].median().idxmax() if "county" in lat.columns else "—"),
                        ("Best State",        lat.groupby("state")["home_value"].median().idxmax()),
                        ("Rental Yield Avg",  f"{(lat['gross_rental_yield'].median()):.2f}%" if lat['gross_rental_yield'].notna().any() else "—"),
                        ("YoY > 5%  ZIPs",    f"{(lat['home_value_yoy_change']>0.05).sum()} ZIPs"),
                        ("YoY < 0%  ZIPs",    f"{(lat['home_value_yoy_change']<0).sum()} ZIPs"),
                    ]]
                ])),
            ]), "1")),
        map_el,
    ], style={"padding":"36px 40px"})

def _stat_row(label, value):
    return html.Div([
        html.Span(label, style={"color":C["muted"],"fontSize":".75rem"}),
        html.Span(value, style={"color":C["text"],"fontSize":".82rem","fontWeight":"600"}),
    ], style={"display":"flex","justifyContent":"space-between",
               "paddingBottom":"9px","marginBottom":"9px",
               "borderBottom":f"1px solid {C['border']}"})

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET PULSE
# ═══════════════════════════════════════════════════════════════════════════════
def render_pulse():
    if MORT.empty:
        return html.Div([
            hdr("Market Pulse","Macro context: mortgage rates & price momentum"),
            notice("Mortgage rate data unavailable. Re-run data_ingestion.py.", "📡"),
        ], style={"padding":"36px 40px"})

    curr_rate = CURR_RATE
    rate_1y   = MORT[MORT["date"] <= MORT["date"].max() - pd.DateOffset(years=1)]["rate"].iloc[-1] \
                if len(MORT) > 52 else None
    rate_delta = curr_rate - rate_1y if rate_1y else None
    med_price  = LATEST["home_value"].median() if not LATEST.empty else None
    payment    = monthly_payment(med_price, curr_rate)

    # ── Mortgage rate history ──
    recent = MORT[MORT["date"] >= "2000-01-01"].copy()
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=recent["date"], y=recent["rate"],
        mode="lines", name="30-Yr Fixed",
        line=dict(color=C["amber"], width=2, shape="spline"),
        fill="tozeroy", fillcolor=rgba(C["amber"], 0.08),
        hovertemplate="%{x|%b %d, %Y}<br>Rate: <b>%{y:.2f}%</b><extra></extra>",
    ))
    for dt, label in [("2008-09-01","2008 Crisis"),("2020-03-01","COVID"),
                       ("2022-03-17","Fed Tightening")]:
        fig_rate.add_vline(x=dt, line_color=C["border2"], line_width=1.5, line_dash="dot")
        fig_rate.add_annotation(x=dt, y=0.95, yref="paper", text=label, showarrow=False,
                                  textangle=-90, font=dict(color=C["muted"],size=9), xanchor="right")
    fig_rate.update_layout(**CHART, height=300)
    fig_rate.update_yaxes(ticksuffix="%")

    # ── Dual axis: DMV price + rate ──
    monthly = DF.groupby("date")["home_value"].median().reset_index()
    mort_m  = MORT.set_index("date").resample("ME")["rate"].mean().reset_index()
    fig_dual = make_subplots(specs=[[{"secondary_y":True}]])
    fig_dual.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["home_value"],
        name="DMV Median Value", mode="lines",
        line=dict(color=C["blue"],width=2.5,shape="spline"),
        fill="tozeroy", fillcolor=rgba(C["blue"],0.07),
        hovertemplate="%{x|%b %Y}<br>Price: <b>$%{y:,.0f}</b><extra></extra>",
    ), secondary_y=False)
    fig_dual.add_trace(go.Scatter(
        x=mort_m["date"], y=mort_m["rate"],
        name="30-Yr Rate", mode="lines",
        line=dict(color=C["amber"],width=2,dash="dash"),
        hovertemplate="%{x|%b %Y}<br>Rate: <b>%{y:.2f}%</b><extra></extra>",
    ), secondary_y=True)
    fig_dual.update_layout(
        **{k:v for k,v in CHART.items() if k not in ("xaxis","yaxis")}, height=340)
    fig_dual.update_yaxes(tickprefix="$", tickformat=",.0f", secondary_y=False,
                           gridcolor="#131313", tickfont=dict(color=C["muted"]))
    fig_dual.update_yaxes(ticksuffix="%", secondary_y=True,
                           tickfont=dict(color=C["amber"]),
                           gridcolor="rgba(0,0,0,0)")

    # ── Rate distribution (recent 5yr) ──
    recent5 = MORT[MORT["date"] >= MORT["date"].max() - pd.DateOffset(years=5)]
    fig_hist = px.histogram(recent5, x="rate", nbins=40,
                             color_discrete_sequence=[C["amber"]],
                             labels={"rate":"30-Yr Rate (%)","count":"Weeks"})
    fig_hist.add_vline(x=curr_rate, line_color=C["red"], line_width=2)
    fig_hist.add_annotation(x=curr_rate, y=1, yref="paper",
                              text=f"  Current: {curr_rate:.2f}%",
                              font=dict(color=C["red"],size=10), showarrow=False)
    fig_hist.update_layout(**CHART, height=260)
    fig_hist.update_traces(marker_line_width=0, opacity=0.8)
    fig_hist.update_xaxes(ticksuffix="%")

    # ── Momentum cards ──
    mom_row = html.Div([
        html.Div([
            html.Div(f"{label} Price Momentum", style={
                "color":C["muted"],"fontSize":".6rem","fontWeight":"700",
                "textTransform":"uppercase","letterSpacing":".1em"}),
            html.Div(
                f"{v:+.2f}%" if v is not None else "—",
                style={"color":C["green"] if (v or 0)>=0 else C["red"],
                       "fontSize":"1.45rem","fontWeight":"800","marginTop":"6px",
                       "letterSpacing":"-.02em"}),
        ], style={
            "background":C["card"],"borderRadius":"10px",
            "border":f"1px solid {C['border']}",
            "borderLeft":f"3px solid {C['green'] if (v or 0)>=0 else C['red']}",
            "padding":"16px 20px","flex":"1",
        }) for label,v in [("1M",MOMENTUM.get("1M")),("3M",MOMENTUM.get("3M")),
                             ("6M",MOMENTUM.get("6M")),("12M",MOMENTUM.get("12M"))]
    ], style={"display":"flex","gap":"10px","marginBottom":"14px"})

    # ── Payment affordability over time ──
    mort_recent = MORT[MORT["date"] >= "2010-01-01"].copy()
    if med_price:
        mort_recent["payment"] = mort_recent["rate"].apply(
            lambda r: monthly_payment(med_price, r))
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(
            x=mort_recent["date"], y=mort_recent["payment"],
            mode="lines", name="Monthly Payment",
            line=dict(color=C["purple"],width=2,shape="spline"),
            fill="tozeroy", fillcolor=rgba(C["purple"],0.08),
            hovertemplate="%{x|%b %Y}<br>Payment: <b>$%{y:,.0f}/mo</b><extra></extra>",
        ))
        fig_pay.add_annotation(
            x=mort_recent["date"].iloc[-1], y=mort_recent["payment"].iloc[-1],
            text=f"  Today: {usd(mort_recent['payment'].iloc[-1])}/mo",
            font=dict(color=C["purple"],size=10), showarrow=False,
        )
        fig_pay.update_layout(**CHART, height=280)
        fig_pay.update_yaxes(tickprefix="$", tickformat=",.0f")
        pay_section = card("Monthly Payment on DMV Median Home · At Each Historical Rate (20% down, 30yr)",
                           g(fig_pay))
    else:
        pay_section = html.Div()

    return html.Div([
        hdr("Market Pulse","Macro context: 30-yr mortgage rates, affordability, and price momentum"),
        row(kpi("30-Yr Rate (Current)", rate(curr_rate), accent=C["amber"],
                delta=f"{rate_delta:+.2f}% vs 1yr ago" if rate_delta else None,
                up=(rate_delta or 0) <= 0),
            kpi("Monthly Payment", usd(payment) if payment else "—",
                delta="on median DMV home · 20% down · 30yr"),
            kpi("Rate vs 2021 Low", f"+{curr_rate - 2.65:+.2f}%" if curr_rate else "—",
                accent=C["red"], up=False),
            kpi("Rate Weeks on Record", f"{len(MORT):,}"),
            gap="10px", mb="20px"),
        mom_row,
        card("30-Year Fixed Mortgage Rate · 2000–Present", g(fig_rate)),
        card("DMV Median Home Value vs 30-Yr Rate (Dual Axis)", g(fig_dual)),
        row(col(card("Rate Distribution · Last 5 Years (red line = today)", g(fig_hist)), "1"),
            col(pay_section or card("Payment History",""), "1")),
    ], style={"padding":"36px 40px"})

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
def render_compare():
    if LATEST.empty:
        return html.Div("No data.", style={"color":C["red"],"padding":"40px"})
    all_zips = sorted(LATEST["zip_code"].astype(str).unique())

    def zip_opt(z):
        city = LATEST[LATEST["zip_code"].astype(str)==z]["city"].values
        label = f"{z}  ·  {city[0]}" if len(city) and city[0] else z
        return {"label": label, "value": z}

    return html.Div([
        hdr("Neighborhood Comparison","Deep dive into any two DMV ZIP codes"),
        row(html.Div([
                html.Div("First ZIP Code", className="input-label"),
                dcc.Dropdown([zip_opt(z) for z in all_zips],
                             all_zips[0] if all_zips else None,
                             id="zip1", clearable=False, className="dark-dropdown"),
            ], style={"flex":"1"}),
            html.Div([
                html.Div("Second ZIP Code", className="input-label"),
                dcc.Dropdown([zip_opt(z) for z in all_zips],
                             all_zips[1] if len(all_zips)>1 else None,
                             id="zip2", clearable=False, className="dark-dropdown"),
            ], style={"flex":"1"})),
        html.Div(id="compare-out"),
    ], style={"padding":"36px 40px"})

@app.callback(Output("compare-out","children"),
              [Input("zip1","value"), Input("zip2","value")])
def update_compare(z1, z2):
    if not z1 or not z2 or str(z1)==str(z2):
        return notice("Select two different ZIP codes to compare.")
    r1 = LATEST[LATEST["zip_code"].astype(str)==str(z1)]
    r2 = LATEST[LATEST["zip_code"].astype(str)==str(z2)]
    if r1.empty or r2.empty:
        return notice("One of the selected ZIPs has no data.", "⚠", C["amber"])
    r1, r2 = r1.iloc[0], r2.iloc[0]
    name1 = f"{r1.get('city',z1)} ({z1})"
    name2 = f"{r2.get('city',z2)} ({z2})"

    p1 = monthly_payment(r1.get("home_value"), CURR_RATE)
    p2 = monthly_payment(r2.get("home_value"), CURR_RATE)

    METRICS = [
        ("Home Value",        "home_value",              usd),
        ("YoY Change",        "home_value_yoy_change",   pct),
        ("Monthly Rent",      "rental_value",            usd),
        ("Gross Rental Yield","gross_rental_yield",      lambda v: f"{v:.2f}%" if pd.notna(v) else "—"),
        ("Median Income",     "median_household_income", usd),
        ("Population",        "total_population",        num),
        ("Unemployment",      "unemployment_rate",       xpct),
        ("Crimes / 1k",       "crimes_per_1000",         lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
        ("Price-to-Income",   "price_to_income_ratio",   lambda v: f"{v:.1f}×" if pd.notna(v) else "—"),
    ]

    def side(row_data, name, color, payment):
        rows = []
        for label, col_name, fmt in METRICS:
            rows.append(html.Div([
                html.Span(label, style={"color":C["muted"],"fontSize":".68rem"}),
                html.Span(fmt(row_data.get(col_name, np.nan)),
                          style={"color":C["text"],"fontSize":".85rem","fontWeight":"600"}),
            ], style={"display":"flex","justifyContent":"space-between",
                       "padding":"7px 0","borderBottom":f"1px solid {C['border']}"}))
        if payment:
            rows.append(html.Div([
                html.Span("Monthly Mortgage", style={"color":C["muted"],"fontSize":".68rem"}),
                html.Span(usd(payment) + "/mo",
                          style={"color":C["purple"],"fontSize":".85rem","fontWeight":"700"}),
            ], style={"display":"flex","justifyContent":"space-between",
                       "padding":"7px 0","borderBottom":f"1px solid {C['border']}"}))
            rows.append(html.Div([
                html.Span(f"Rate assumption: {CURR_RATE:.2f}%, 20% down, 30yr",
                          style={"color":C["dim"],"fontSize":".65rem","marginTop":"6px"}),
            ]))
        return html.Div([
            html.Div(name, style={"color":color,"fontWeight":"700","fontSize":".9rem",
                                   "marginBottom":"14px","paddingBottom":"12px",
                                   "borderBottom":f"1px solid {C['border']}"}),
            *rows,
        ], style={
            "flex":"1","background":C["card"],"borderRadius":"10px",
            "border":f"1px solid {C['border']}","borderTop":f"2px solid {color}","padding":"22px",
        })

    # Price history
    trend = DF[DF["zip_code"].astype(str).isin([str(z1),str(z2)])].copy()
    trend["label"] = trend["zip_code"].astype(str).map({str(z1):name1, str(z2):name2})
    fig_t = px.line(trend, x="date", y="home_value", color="label",
                    color_discrete_sequence=[C["blue"],C["red"]],
                    labels={"home_value":"Value ($)","date":"","label":""})
    fig_t.update_traces(line_width=2.5, line_shape="spline")
    fig_t.update_layout(**CHART, height=300)
    fig_t.update_yaxes(tickprefix="$", tickformat=",.0f")

    # Rental history (if available)
    rent_figs = []
    if trend["rental_value"].notna().any():
        fig_r = px.line(trend, x="date", y="rental_value", color="label",
                        color_discrete_sequence=[C["purple"],C["amber"]],
                        labels={"rental_value":"Rent ($/mo)","date":"","label":""})
        fig_r.update_traces(line_width=2, line_shape="spline")
        fig_r.update_layout(**CHART, height=260)
        fig_r.update_yaxes(tickprefix="$", tickformat=",.0f")
        rent_figs = [col(card("Rental Value History", g(fig_r)),"1")]

    # Radar
    rcols = ["home_value","median_household_income","total_population","college_educated_count"]
    rlabels = ["Home Value","Income","Population","Educated"]
    avail = [c for c in rcols if c in LATEST.columns]
    fig_rad = go.Figure()
    for rdata, name, color in [(r1,name1,C["blue"]),(r2,name2,C["red"])]:
        vals = [rdata.get(c,0) or 0 for c in avail]
        maxv = [LATEST[c].max() for c in avail]
        norm = [v/mx if mx else 0 for v,mx in zip(vals,maxv)]
        norm.append(norm[0])
        lbls = [rlabels[i] for i in range(len(avail))] + [rlabels[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=norm, theta=lbls, name=name,
            line=dict(color=color,width=2), fill="toself", opacity=0.3,
        ))
    fig_rad.update_layout(
        **{k:v for k,v in CHART.items() if k not in ("xaxis","yaxis")}, height=300,
        polar=dict(bgcolor="#080808",
                   radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color=C["muted"])),
                   angularaxis=dict(tickfont=dict(color=C["muted"]))))

    return html.Div([
        row(side(r1, name1, C["blue"], p1), side(r2, name2, C["red"], p2)),
        row(col(card("Price History", g(fig_t)),"3"),
            col(card("Relative Comparison", g(fig_rad)),"2")),
        *([row(*rent_figs)] if rent_figs else []),
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
def render_predictor():
    if not HAS_CENSUS:
        return html.Div([
            hdr("Price Predictor","Machine learning home value estimator"),
            html.Div([
                html.Div("🔑", style={"fontSize":"2.2rem","marginBottom":"12px"}),
                html.Div("Census Data Required", style={"color":C["text"],"fontWeight":"800",
                                                          "fontSize":"1.1rem","marginBottom":"10px"}),
                html.Div([
                    "The ML model needs Census ACS features (income, population, unemployment). ",
                    html.Br(), html.Br(),
                    "Get a free key at ",
                    html.A("api.census.gov/data/key_signup.html",
                           href="https://api.census.gov/data/key_signup.html",
                           target="_blank", style={"color":C["blue"]}),
                    ", add to .env as ", html.Code("CENSUS_API_KEY=...",
                    style={"background":C["card2"],"padding":"2px 7px","borderRadius":"4px",
                           "fontSize":".8rem"}),
                    ", then re-run the pipeline.",
                ], style={"color":C["muted"],"fontSize":".87rem","lineHeight":"1.8"}),
            ], style={"background":C["card"],"border":f"1px solid {C['border']}",
                       "borderLeft":f"3px solid {C['amber']}","borderRadius":"10px",
                       "padding":"30px 34px","maxWidth":"520px"}),
        ], style={"padding":"36px 40px"})

    def inp(label, id_, val, mn, mx, step, unit=""):
        return html.Div([
            html.Div(label + (f"  ({unit})" if unit else ""), style={
                "color":C["muted"],"fontSize":".62rem","fontWeight":"700",
                "textTransform":"uppercase","letterSpacing":".08em","marginBottom":"5px"}),
            dcc.Input(id=id_, type="number", value=val, min=mn, max=mx, step=step,
                      className="dark-input"),
        ], style={"marginBottom":"12px"})

    return html.Div([
        hdr("Price Predictor","Enter neighbourhood characteristics to estimate market value"),
        row(col(html.Div([
                card("Demographics", html.Div([
                    inp("Median Household Income", "p-income", 105000, 30000, 400000, 5000, "$"),
                    inp("ZIP Code Population",     "p-pop",    18000,  100,   120000, 500),
                    inp("Unemployment Rate",       "p-unemp",  4.5,    0.0,   25.0,   0.5, "%"),
                    inp("College-Educated",        "p-edu",    6000,   0,     60000,  200),
                ])),
                card("Housing & Safety", html.Div([
                    inp("Crimes per 1,000",  "p-crime", 22.0, 0.0, 120.0, 1.0),
                    inp("Median Year Built", "p-yr",    1988, 1935, 2024,  1),
                    inp("Median Gross Rent", "p-rent",  1950, 500,  6000,  50, "$/mo"),
                ])),
                html.Button("Predict →", id="pred-btn", n_clicks=0, style={
                    "width":"100%","padding":"13px","background":C["blue"],
                    "border":"none","borderRadius":"8px","color":"#fff",
                    "fontSize":".95rem","fontWeight":"700","cursor":"pointer",
                }),
            ]), flex="1", ),
            col(html.Div(id="pred-out",
                         children=notice("Fill the form and click Predict →")),
                flex="2"),
            gap="24px"),
    ], style={"padding":"36px 40px"})

@app.callback(
    Output("pred-out","children"), Input("pred-btn","n_clicks"),
    [State("p-income","value"), State("p-pop","value"), State("p-unemp","value"),
     State("p-edu","value"), State("p-crime","value"), State("p-yr","value"),
     State("p-rent","value")],
    prevent_initial_call=True,
)
def do_predict(_, income, pop, unemp, edu, crime, yr, rent):
    vals = [income, pop, unemp, edu, crime, yr, rent]
    if any(v is None for v in vals):
        return notice("Fill all fields before predicting.", "⚠", C["amber"])
    bundle = load_model()
    if bundle:
        model, scaler = bundle["model"], bundle.get("scaler")
    else:
        train = LATEST[FEATURES+["home_value"]].dropna()
        if train.empty:
            return notice("No training data. Load Census data first.", "⚠", C["red"])
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(train[FEATURES], train["home_value"])
        scaler = None
    inp_df = pd.DataFrame([vals], columns=FEATURES)
    X = scaler.transform(inp_df) if scaler else inp_df
    pred = model.predict(X)[0]
    lo = hi = None
    if hasattr(model,"estimators_"):
        tp = np.array([t.predict(inp_df)[0] for t in model.estimators_])
        lo, hi = np.percentile(tp, [10,90])
    pay = monthly_payment(pred, CURR_RATE) if CURR_RATE else None
    base = LATEST[FEATURES+["home_value","zip_code","city","county"]].dropna(subset=FEATURES).copy()
    base["_d"] = ((base[FEATURES].values - inp_df.values)**2).sum(axis=1)
    sim = base.nsmallest(5,"_d")[["zip_code","city","county","home_value"]].reset_index(drop=True)
    sim["home_value"] = sim["home_value"].apply(usd)
    return html.Div([
        html.Div([
            html.Div("Predicted Market Value", style={"color":C["muted"],"fontSize":".62rem",
                      "fontWeight":"700","textTransform":"uppercase","letterSpacing":".1em"}),
            html.Div(usd(pred), style={"color":C["blue"],"fontSize":"2.8rem","fontWeight":"900",
                                        "letterSpacing":"-.04em","marginTop":"8px"}),
            html.Div(f"80% interval: {usd(lo)} – {usd(hi)}" if lo else "",
                     style={"color":C["muted"],"fontSize":".8rem","marginTop":"4px"}),
            *([html.Div(f"Est. monthly payment: {usd(pay)}/mo · {CURR_RATE:.2f}%, 20% down, 30yr",
                        style={"color":C["purple"],"fontSize":".82rem","marginTop":"8px",
                               "fontWeight":"500"})] if pay else []),
        ], style={"background":C["card"],"border":f"1px solid {C['border']}",
                   "borderTop":f"2px solid {C['blue']}","borderRadius":"10px",
                   "padding":"26px","marginBottom":"14px"}),
        card("Most Similar Neighbourhoods", dash_table.DataTable(
            data=sim.to_dict("records"),
            columns=[{"name":c.replace("_"," ").title(),"id":c} for c in sim.columns],
            style_table={"overflowX":"auto"},
            style_header=_tbl_header(), style_cell=_tbl_cell(),
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#0a0a0a"}],
        )),
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HIDDEN GEMS
# ═══════════════════════════════════════════════════════════════════════════════
def render_gems():
    if os.path.exists(UV_PATH):
        uv = pd.read_csv(UV_PATH)
    elif not LATEST.empty and HAS_CENSUS:
        bundle = load_model()
        if not bundle:
            return html.Div([
                hdr("Hidden Gems","Statistically undervalued ZIP codes"),
                notice("Run python src/analysis.py to generate undervalued ZIPs.", "💎"),
            ], style={"padding":"36px 40px"})
        model = bundle["model"]; scaler = bundle.get("scaler")
        base = LATEST.dropna(subset=FEATURES+["home_value"]).copy()
        X = scaler.transform(base[FEATURES]) if scaler else base[FEATURES]
        base["predicted_value"] = model.predict(X)
        base["value_gap"] = base["predicted_value"] - base["home_value"]
        base["value_gap_pct"] = (base["value_gap"] / base["home_value"] * 100).round(1)
        from scipy import stats as sp
        base["gap_zscore"] = sp.zscore(base["value_gap"])
        uv = base[base["gap_zscore"] > 1.0].copy()
    else:
        return html.Div([
            hdr("Hidden Gems","Statistically undervalued ZIP codes"),
            notice("Census data required. Add CENSUS_API_KEY to .env and re-run the pipeline.", "💎"),
        ], style={"padding":"36px 40px"})

    if uv.empty:
        return html.Div([hdr("Hidden Gems"),
                          notice("No undervalued ZIPs identified.")],
                         style={"padding":"36px 40px"})

    uv = uv.sort_values("value_gap_pct", ascending=False).head(20)
    lbl = "city" if "city" in uv.columns else "zip_code"
    fig = px.bar(uv.head(15).sort_values("value_gap_pct"),
                 x="value_gap_pct", y=lbl, orientation="h", color="value_gap_pct",
                 color_continuous_scale=[[0,"#0a2e18"],[1,C["green"]]],
                 text="value_gap_pct",
                 labels={"value_gap_pct":"Predicted Upside (%)",lbl:""})
    fig.update_traces(texttemplate="%{text:+.1f}%", textposition="outside",
                       textfont=dict(color=C["muted"],size=10), marker_line_width=0)
    fig.update_layout(**CHART, height=480)
    fig.update_coloraxes(showscale=False)
    show = [c for c in ["zip_code","city","county","home_value","predicted_value","value_gap_pct"]
            if c in uv.columns]
    return html.Div([
        hdr("Hidden Gems","ZIPs where the model predicts higher values than current market price"),
        card("Predicted Upside vs Current Market", g(fig)),
        card("Ranked Opportunities", dash_table.DataTable(
            data=uv[show].round(1).to_dict("records"),
            columns=[{"name":c.replace("_"," ").title(),"id":c} for c in show],
            sort_action="native",
            style_table={"overflowX":"auto"},
            style_header=_tbl_header(), style_cell=_tbl_cell(),
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#0a0a0a"}],
        )),
    ], style={"padding":"36px 40px"})

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
def render_explorer():
    if LATEST.empty:
        return html.Div("No data.", style={"color":C["red"],"padding":"40px"})
    COLS = [c for c in ["zip_code","city","county","state","home_value",
                         "home_value_yoy_change","rental_value","gross_rental_yield",
                         "median_household_income","total_population",
                         "unemployment_rate","crimes_per_1000",
                         "price_to_income_ratio","latitude","longitude"] if c in LATEST.columns]
    display = LATEST[COLS].copy()
    if "home_value_yoy_change" in display.columns:
        display["home_value_yoy_change"] = (display["home_value_yoy_change"]*100).round(2)

    # Distribution strip for home values
    fig_strip = px.histogram(
        display, x="home_value", nbins=40,
        color_discrete_sequence=[C["blue"]],
        labels={"home_value":"Home Value ($)"},
    )
    fig_strip.update_layout(**CHART, height=180)
    fig_strip.update_layout(margin=dict(l=8,r=8,t=8,b=8))
    fig_strip.update_traces(marker_line_width=0, opacity=0.8)
    fig_strip.update_xaxes(tickprefix="$", tickformat=",.0f")

    return html.Div([
        hdr("Data Explorer", f"{len(display):,} ZIP codes — sortable, filterable, CSV export"),
        row(col(card("Home Value Distribution", g(fig_strip)), "2"),
            col(html.Div([
                card("Snapshot Stats", html.Div([
                    _stat_row("ZIP Codes", num(len(display))),
                    _stat_row("Median Value", usd(display["home_value"].median())),
                    _stat_row("Mean Value", usd(display["home_value"].mean())),
                    _stat_row("Max Value", usd(display["home_value"].max())),
                    _stat_row("Min Value", usd(display["home_value"].min())),
                    *([_stat_row("YoY Positive", f"{(display['home_value_yoy_change']>0).sum()} ZIPs")]
                      if "home_value_yoy_change" in display.columns else []),
                ]), mb="0px"),
            ]), "1")),
        card("Full Dataset", dash_table.DataTable(
            id="explorer-tbl",
            data=display.round(2).to_dict("records"),
            columns=[{"name":c.replace("_"," ").title(),"id":c} for c in COLS],
            sort_action="native", filter_action="native",
            page_size=30, page_action="native", export_format="csv",
            style_table={"overflowX":"auto"},
            style_header=_tbl_header(),
            style_cell=_tbl_cell(),
            style_data_conditional=[
                {"if":{"row_index":"odd"},"background":"#0a0a0a"},
                {"if":{"filter_query":"{home_value_yoy_change} > 0",
                        "column_id":"home_value_yoy_change"},"color":C["green"]},
                {"if":{"filter_query":"{home_value_yoy_change} < 0",
                        "column_id":"home_value_yoy_change"},"color":C["red"]},
            ],
        )),
    ], style={"padding":"36px 40px"})

# ── shared table styles ───────────────────────────────────────────────────────
def _tbl_header():
    return {"background":"#080808","color":C["muted"],"fontWeight":"700",
             "fontSize":".62rem","border":f"1px solid {C['border']}",
             "textTransform":"uppercase","letterSpacing":".08em","padding":"11px 14px"}

def _tbl_cell():
    return {"background":C["card"],"color":C["text"],"border":f"1px solid {C['border']}",
             "fontSize":".84rem","padding":"10px 14px","fontFamily":"Inter,sans-serif"}

# ── router ────────────────────────────────────────────────────────────────────
@app.callback(Output("main","children"), Input("page","data"))
def route(p):
    if p=="overview":  return render_overview()
    if p=="pulse":     return render_pulse()
    if p=="compare":   return render_compare()
    if p=="predictor": return render_predictor()
    if p=="gems":      return render_gems()
    if p=="explorer":  return render_explorer()
    return render_overview()

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")
