from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent
FINANCIALS_PATH = ROOT / "data" / "nike_financials_official.csv"
DEMO_PATH = ROOT / "data" / "product_sales_demo_synthetic.csv"

st.set_page_config(
    page_title="Nike | AI Sales Intelligence",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #101010;
        --volt: #D7FF00;
        --muted: #6B7280;
        --line: #E8E8E8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F5F5F2;
        color: #101010;
    }

    [data-testid="stHeader"] {
        background: rgba(245,245,242,.96);
    }

    [data-testid="stSidebar"] {
        background: #101010;
    }

    [data-testid="stSidebar"] * {
        color: #F8F8F6 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border-color: #555;
    }

    [data-testid="stSidebar"] .stButton button {
        background: #D7FF00;
        color: #111 !important;
        border: 0;
        font-weight: 700;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
        color: #101010;
    }

    h1 {
        font-size: 2.6rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
    }

    .eyebrow {
        font-size: .72rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        font-weight: 700;
        color: #555;
    }

    .hero {
        background: #101010;
        color: #fff;
        padding: 2rem 2.2rem;
        border-radius: 22px;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }

    .hero:after {
        content: 'NIKE';
        position: absolute;
        right: 1.6rem;
        bottom: -2.3rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 8rem;
        letter-spacing: -.1em;
        color: rgba(255,255,255,.055);
        transform: skew(-12deg);
    }

    .hero-kicker {
        color: #D7FF00;
        font-size: .75rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: .6rem;
    }

    .hero-title {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -.05em;
        max-width: 720px;
    }

    .hero-copy {
        color: #D0D0D0;
        font-size: .96rem;
        margin-top: .8rem;
        max-width: 690px;
    }

    .hero-pill {
        display: inline-block;
        margin-top: 1rem;
        padding: .38rem .65rem;
        border-radius: 999px;
        background: #D7FF00;
        color: #111;
        font-size: .72rem;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #EAEAE7;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 3px 14px rgba(0,0,0,.025);
    }

    [data-testid="stMetricLabel"] {
        color: #454545 !important;
        font-size: .82rem;
        font-weight: 700;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #101010 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #176B35 !important;
    }

    [data-testid="stMarkdown"] p,
    [data-testid="stCaptionContainer"] p {
        color: #333333;
    }

    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button p {
        color: #343434 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #101010 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -.04em;
    }

    [data-testid="stMetricDelta"] {
        font-size: .78rem;
    }

    .section-head {
        display: flex;
        align-items: center;
        gap: .65rem;
        margin: 1.1rem 0 .3rem;
    }

    .section-number {
        background: #D7FF00;
        color: #111;
        border-radius: 8px;
        padding: .25rem .5rem;
        font-size: .75rem;
        font-weight: 800;
    }

    .section-note {
        color: #6B7280;
        font-size: .88rem;
        margin-bottom: 1rem;
    }

    .insight-card {
        background: #fff;
        border: 1px solid #E7E7E3;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        min-height: 112px;
    }

    .insight-label {
        color: #4B4B4B !important;
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-weight: 700;
    }

    .insight-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.22rem;
        font-weight: 700;
        margin: .25rem 0;
        color: #111111 !important;
        overflow-wrap: anywhere;
    }

    .insight-copy {
        color: #454545 !important;
        font-size: .82rem;
    }

    .note {
        background: #F0F3E0;
        border-left: 4px solid #B7D600;
        padding: .85rem 1rem;
        border-radius: 0 10px 10px 0;
        font-size: .86rem;
        color: #222;
    }

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #DEDEDA;
        color: #777;
        font-size: .75rem;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 700;
    }

    /* High-contrast action and download buttons */
    .stDownloadButton button,
    .stButton button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-tertiary"] {
        background-color: #D7FF00 !important;
        color: #111111 !important;
        border: 1px solid #A8C400 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    .stDownloadButton button *,
    .stButton button *,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-tertiary"] * {
        color: #111111 !important;
        fill: #111111 !important;
    }

    /* Keep the sidebar expand/collapse arrow visible */
    [data-testid="stHeader"] button,
    [data-testid="collapsedControl"] button,
    button[kind="header"] {
        color: #111111 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D0D0D0 !important;
        opacity: 1 !important;
    }

    [data-testid="stHeader"] button svg,
    [data-testid="collapsedControl"] button svg,
    button[kind="header"] svg {
        color: #111111 !important;
        fill: #111111 !important;
        stroke: #111111 !important;
        opacity: 1 !important;
    }

    /*
    THREE-DOT MENU FIX
    Force the menu popup and all nested labels to use a light
    background with dark text. These selectors cover Streamlit
    and Base Web menu/popover implementations.
    */

    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] button,
    [data-testid="stToolbar"] button {
        color: #111111 !important;
        background-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stMainMenu"] svg,
    [data-testid="stToolbar"] svg {
        color: #111111 !important;
        fill: #111111 !important;
        stroke: #111111 !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [role="menu"],
    [role="listbox"],
    [data-testid="stPopoverBody"] {
        background: #FFFFFF !important;
        color: #111111 !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [role="menu"] *,
    [role="menuitem"],
    [role="menuitem"] *,
    [role="option"],
    [role="option"] *,
    [data-testid="stPopoverBody"] * {
        color: #111111 !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }

    [role="menuitem"],
    [role="option"] {
        background-color: #FFFFFF !important;
    }

    [role="menuitem"]:hover,
    [role="option"]:hover,
    [data-baseweb="menu"] li:hover {
        background-color: #EDEFE7 !important;
        color: #111111 !important;
    }

    [data-baseweb="popover"] svg,
    [data-baseweb="menu"] svg,
    [role="menu"] svg {
        color: #111111 !important;
        fill: #111111 !important;
        stroke: #111111 !important;
    }

    /* Keep chart labels readable */
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .legend text,
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .axis-title {
        fill: #222222 !important;
        color: #222222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def usd_m(value):
    return f"${value / 1000:,.2f}B"


def plot_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans, sans-serif",
            color="#222222",
            size=12,
        ),
        title=dict(
            font=dict(
                family="Space Grotesk, sans-serif",
                size=16,
                color="#101010",
            ),
            x=0.02,
            xanchor="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            title=None,
        ),
        hoverlabel=dict(
            bgcolor="#101010",
            font_color="#FFFFFF",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#B8B8B8",
        zeroline=False,
        tickfont=dict(color="#303030", size=12),
        title_font=dict(color="#222222", size=13),
        automargin=True,
    )

    fig.update_yaxes(
        gridcolor="#E5E7EB",
        zeroline=False,
        linecolor="#B8B8B8",
        tickfont=dict(color="#303030", size=12),
        title_font=dict(color="#222222", size=13),
        automargin=True,
    )

    return fig


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="font-family:Space Grotesk;font-size:1.3rem;
        font-weight:700;letter-spacing:-.04em">
        NIKE <span style="color:#D7FF00">/</span> INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("MLPA · Sales analytics prototype")
    st.divider()

    st.markdown("**DATA SOURCE**")

    uploaded = st.file_uploader(
        "Upload product-level CSV",
        type=["csv"],
        help="Required columns: date, product_model, units_sold, revenue_usd",
    )

    st.caption(
        "Company financials use reported annual data. Product-level "
        "demo data is synthetic unless you upload a verified dataset."
    )

    st.divider()
    st.markdown("**NAVIGATION**")
    st.caption(
        "Scroll through company performance, product sales, forecasting, "
        "and recommendations."
    )

    st.divider()

    st.markdown(
        """
        <div style="font-size:.75rem;color:#BDBDBD">
        DESIGNED FOR<br>
        <b style="color:#FFFFFF">MLPA PROJECT REVIEW</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            Performance intelligence · FY2024—FY2026
        </div>
        <div class="hero-title">
            Read the numbers.<br>Spot the next move.
        </div>
        <div class="hero-copy">
            An interactive sales intelligence workspace combining
            reported company financials, product-level analysis,
            and an educational machine-learning forecast.
        </div>
        <span class="hero-pill">AI SALES INTELLIGENCE</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# COMPANY FINANCIAL DATA
# ---------------------------------------------------------

if not FINANCIALS_PATH.exists():
    st.error(
        "Financial CSV not found. Make sure "
        "data/nike_financials_official.csv exists inside the project folder."
    )
    st.stop()

fin = pd.read_csv(FINANCIALS_PATH)

needed_fin = {
    "fiscal_year",
    "nike_inc_revenue_usd_m",
    "nike_brand_footwear_usd_m",
    "nike_brand_apparel_usd_m",
    "nike_brand_equipment_usd_m",
    "nike_direct_usd_m",
    "nike_wholesale_usd_m",
}

if not needed_fin.issubset(fin.columns):
    st.error(
        "The financial CSV columns do not match the expected template. "
        "Restore the original header row and try again."
    )
    st.stop()

fin = fin.sort_values("fiscal_year").copy()
latest = fin.iloc[-1]
prior = fin.iloc[-2] if len(fin) > 1 else None

rev_growth = (
    (latest.nike_inc_revenue_usd_m / prior.nike_inc_revenue_usd_m - 1) * 100
    if prior is not None and prior.nike_inc_revenue_usd_m
    else np.nan
)

foot_growth = (
    (latest.nike_brand_footwear_usd_m / prior.nike_brand_footwear_usd_m - 1) * 100
    if prior is not None and prior.nike_brand_footwear_usd_m
    else np.nan
)

app_growth = (
    (latest.nike_brand_apparel_usd_m / prior.nike_brand_apparel_usd_m - 1) * 100
    if prior is not None and prior.nike_brand_apparel_usd_m
    else np.nan
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    f"{latest.fiscal_year} revenue",
    usd_m(latest.nike_inc_revenue_usd_m),
    f"{rev_growth:+.2f}% vs prior FY" if pd.notna(rev_growth) else None,
)

k2.metric(
    "Nike Brand footwear",
    usd_m(latest.nike_brand_footwear_usd_m),
    f"{foot_growth:+.2f}% vs prior FY" if pd.notna(foot_growth) else None,
)

k3.metric(
    "Nike Brand apparel",
    usd_m(latest.nike_brand_apparel_usd_m),
    f"{app_growth:+.2f}% vs prior FY" if pd.notna(app_growth) else None,
)

k4.metric("Financial periods", f"{len(fin)} years", "Reported annual data")


# ---------------------------------------------------------
# PRODUCT DATA
# ---------------------------------------------------------

if uploaded is not None:
    try:
        sales = pd.read_csv(uploaded)
        data_label = (
            "UPLOADED DATASET · Verify source, coverage, and brand before interpreting"
        )
        is_demo = False
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    if not DEMO_PATH.exists():
        st.error(
            "Product demo CSV not found. Make sure "
            "data/product_sales_demo_synthetic.csv exists."
        )
        st.stop()

    sales = pd.read_csv(DEMO_PATH)
    data_label = "SYNTHETIC DEMO DATA · Not actual Nike model sales"
    is_demo = True

required = {"date", "product_model", "units_sold", "revenue_usd"}

if not required.issubset(sales.columns):
    st.error(
        f"Product CSV is missing required columns: "
        f"{', '.join(sorted(required - set(sales.columns)))}"
    )
    st.stop()

sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="coerce")
sales["revenue_usd"] = pd.to_numeric(sales["revenue_usd"], errors="coerce")
sales["product_model"] = sales["product_model"].astype(str)

sales = sales.dropna(
    subset=["date", "product_model", "units_sold", "revenue_usd"]
).copy()

if sales.empty:
    st.error(
        "The product dataset contains no valid rows after parsing. "
        "Check date, units_sold, and revenue_usd columns."
    )
    st.stop()

sales["month"] = sales["date"].dt.to_period("M").dt.to_timestamp()
sales["year"] = sales["date"].dt.year

monthly = sales.groupby(
    ["month", "product_model"], as_index=False
).agg(
    units_sold=("units_sold", "sum"),
    revenue_usd=("revenue_usd", "sum"),
)

rank = sales.groupby(
    "product_model", as_index=False
).agg(
    total_revenue_usd=("revenue_usd", "sum"),
    total_units=("units_sold", "sum"),
).sort_values("total_revenue_usd", ascending=False)

st.markdown(
    f"<div class='note'><b>DATA TRANSPARENCY</b> &nbsp; {data_label}. "
    "Company financials and product-level model sales are different "
    "datasets; do not treat them as directly comparable.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='note'><b>ONE-PAGE VIEW</b> &nbsp; Scroll through company "
    "performance, product rankings, machine-learning forecast, and "
    "business actions below.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SECTION 1: COMPANY PERFORMANCE
# ---------------------------------------------------------

st.markdown(
    "<div class='section-head'><span class='section-number'>01</span>"
    "<h2>Company performance</h2></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-note'>Reported annual financials · "
    "USD billions unless noted otherwise</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1])

with left:
    revenue_long = fin.melt(
        id_vars="fiscal_year",
        value_vars=[
            "nike_inc_revenue_usd_m",
            "nike_brand_footwear_usd_m",
            "nike_brand_apparel_usd_m",
        ],
        var_name="metric",
        value_name="usd_m",
    )

    metric_names = {
        "nike_inc_revenue_usd_m": "NIKE, Inc. revenue",
        "nike_brand_footwear_usd_m": "Nike Brand footwear",
        "nike_brand_apparel_usd_m": "Nike Brand apparel",
    }

    revenue_long["metric"] = revenue_long["metric"].map(metric_names)

    fig = px.line(
        revenue_long,
        x="fiscal_year",
        y="usd_m",
        color="metric",
        markers=True,
        title="Revenue trend by fiscal year",
        color_discrete_sequence=["#111111", "#B2D600", "#8B8B8B"],
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_yaxes(title="USD millions")
    st.plotly_chart(plot_layout(fig, 390), use_container_width=True)

with right:
    channel = pd.DataFrame(
        {
            "Channel": ["Nike Direct", "Wholesale"],
            "USD millions": [
                latest.nike_direct_usd_m,
                latest.nike_wholesale_usd_m,
            ],
        }
    )

    fig = px.bar(
        channel,
        x="USD millions",
        y="Channel",
        orientation="h",
        title=f"{latest.fiscal_year} channel mix",
        text="USD millions",
        color="Channel",
        color_discrete_map={
            "Nike Direct": "#B2D600",
            "Wholesale": "#111111",
        },
    )

    fig.update_traces(
        texttemplate="$%{x:,.0f}M",
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
    )

    fig.update_xaxes(title="USD millions")
    st.plotly_chart(plot_layout(fig, 390), use_container_width=True)

st.markdown("**Reported financial data**")

display_fin = fin[
    [
        "fiscal_year",
        "nike_inc_revenue_usd_m",
        "nike_brand_footwear_usd_m",
        "nike_brand_apparel_usd_m",
        "nike_brand_equipment_usd_m",
        "nike_direct_usd_m",
        "nike_wholesale_usd_m",
    ]
].copy()

display_fin.columns = [
    "Fiscal year",
    "NIKE, Inc. revenue ($M)",
    "Footwear ($M)",
    "Apparel ($M)",
    "Equipment ($M)",
    "Nike Direct ($M)",
    "Wholesale ($M)",
]

st.dataframe(display_fin, use_container_width=True, hide_index=True)

st.caption(
    "Financial categories and sales channels have different reporting scopes; "
    "channel values should not be added to category values. See the linked "
    "annual report in Sources."
)


# ---------------------------------------------------------
# SECTION 2: PRODUCT INTELLIGENCE
# ---------------------------------------------------------

st.markdown(
    "<div class='section-head'><span class='section-number'>02</span>"
    "<h2>Product intelligence</h2></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-note'>Rank products by revenue and units sold "
    "within the loaded dataset—not a verified global Nike shoe ranking. "
    "Revenue leader and unit leader can be different models.</div>",
    unsafe_allow_html=True,
)

date_min = sales.date.min().date()
date_max = sales.date.max().date()

range_value = st.date_input(
    "Analysis period",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

filtered = sales.copy()

if isinstance(range_value, (tuple, list)) and len(range_value) == 2:
    filtered = filtered[
        (filtered.date.dt.date >= range_value[0])
        & (filtered.date.dt.date <= range_value[1])
    ]

if filtered.empty:
    st.warning("No rows in this date range. Select a wider analysis period.")
else:
    product_rank = filtered.groupby(
        "product_model", as_index=False
    ).agg(
        total_revenue_usd=("revenue_usd", "sum"),
        total_units=("units_sold", "sum"),
    ).sort_values("total_revenue_usd", ascending=False)

    p1, p2, p3 = st.columns(3)
    p1.metric("Revenue in period", f"${filtered.revenue_usd.sum():,.0f}")
    p2.metric("Units in period", f"{filtered.units_sold.sum():,.0f}")
    p3.metric("Products compared", f"{filtered.product_model.nunique()}")

    best = product_rank.sort_values(
        "total_revenue_usd", ascending=False
    ).iloc[0]

    low = product_rank.sort_values(
        "total_revenue_usd", ascending=True
    ).iloc[0]

    most_units = product_rank.sort_values(
        "total_units", ascending=False
    ).iloc[0]

    least_units = product_rank.sort_values(
        "total_units", ascending=True
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div class='insight-card'>"
            f"<div class='insight-label'>🏆 Highest revenue</div>"
            f"<div class='insight-value'>{best.product_model}</div>"
            f"<div class='insight-copy'>"
            f"${best.total_revenue_usd:,.0f} revenue · "
            f"{best.total_units:,.0f} units</div></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div class='insight-card'>"
            f"<div class='insight-label'>🔥 Most units sold</div>"
            f"<div class='insight-value'>{most_units.product_model}</div>"
            f"<div class='insight-copy'>"
            f"{most_units.total_units:,.0f} units · "
            f"${most_units.total_revenue_usd:,.0f} revenue</div></div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"<div class='insight-card'>"
            f"<div class='insight-label'>📉 Fewest units sold</div>"
            f"<div class='insight-value'>{least_units.product_model}</div>"
            f"<div class='insight-copy'>"
            f"{least_units.total_units:,.0f} units · "
            f"${least_units.total_revenue_usd:,.0f} revenue</div></div>",
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"<div class='insight-card'>"
            f"<div class='insight-label'>Lowest revenue</div>"
            f"<div class='insight-value'>{low.product_model}</div>"
            f"<div class='insight-copy'>"
            f"${low.total_revenue_usd:,.0f} revenue · "
            f"{low.total_units:,.0f} units</div></div>",
            unsafe_allow_html=True,
        )

    chart_left, chart_right = st.columns([1.25, 1])

    with chart_left:
        top = product_rank.head(10).sort_values("total_revenue_usd")

        fig = px.bar(
            top,
            x="total_revenue_usd",
            y="product_model",
            orientation="h",
            title="Top products by revenue",
            text="total_revenue_usd",
            color="total_revenue_usd",
            color_continuous_scale=[
                [0, "#B2D600"],
                [1, "#111111"],
            ],
        )

        fig.update_traces(
            texttemplate="$%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="Revenue (USD)")
        st.plotly_chart(plot_layout(fig, 420), use_container_width=True)

    with chart_right:
        fig = px.scatter(
            product_rank,
            x="total_units",
            y="total_revenue_usd",
            text="product_model",
            title="Units vs revenue",
            color_discrete_sequence=["#111111"],
        )

        fig.update_traces(
            marker=dict(
                size=12,
                color="#B2D600",
                line=dict(color="#111111", width=1),
            ),
            textposition="top center",
        )

        fig.update_xaxes(title="Units sold")
        fig.update_yaxes(title="Revenue (USD)")
        st.plotly_chart(plot_layout(fig, 420), use_container_width=True)

    st.markdown("**Product ranking**")

    display_rank = product_rank.copy()
    display_rank.columns = [
        "Product model",
        "Revenue (USD)",
        "Units sold",
    ]

    st.dataframe(display_rank, use_container_width=True, hide_index=True)

    # Allow the user to export product rankings.
    st.download_button(
        "Download product ranking CSV",
        data=display_rank.to_csv(index=False).encode("utf-8"),
        file_name="nike_product_ranking.csv",
        mime="text/csv",
    )

    yearly = filtered.groupby(
        ["year", "product_model"], as_index=False
    ).agg(
        revenue_usd=("revenue_usd", "sum"),
        units_sold=("units_sold", "sum"),
    )

    years = sorted(yearly.year.unique())

    if len(years) >= 2:
        y_prev, y_last = years[-2], years[-1]

        comp = yearly[
            yearly.year.isin([y_prev, y_last])
        ].pivot(
            index="product_model",
            columns="year",
            values="revenue_usd",
        ).fillna(0)

        comp["Change %"] = np.where(
            comp[y_prev] != 0,
            (comp[y_last] / comp[y_prev] - 1) * 100,
            np.nan,
        )

        comp = comp.reset_index().rename(
            columns={
                y_prev: f"Revenue {y_prev} ($)",
                y_last: f"Revenue {y_last} ($)",
            }
        ).sort_values("Change %", ascending=False)

        st.markdown("**Year-over-year comparison**")

        st.caption(
            f"Calendar-year comparison: {y_last} vs {y_prev}; "
            "this is not Nike fiscal-year reporting."
        )

        st.dataframe(comp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# SECTION 3: MACHINE-LEARNING FORECAST
# ---------------------------------------------------------

st.markdown(
    "<div class='section-head'><span class='section-number'>03</span>"
    "<h2>Machine-learning forecast</h2></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-note'>Random Forest regression with lag, "
    "rolling-average, calendar, and time features.</div>",
    unsafe_allow_html=True,
)

available_products = sorted(monthly.product_model.unique())

selected_product = st.selectbox(
    "Select product",
    available_products,
)

horizon = st.slider(
    "Forecast horizon",
    min_value=1,
    max_value=6,
    value=3,
    format="%d months",
)

p = (
    monthly[monthly.product_model == selected_product]
    .sort_values("month")
    .set_index("month")
    .asfreq("MS")
)

p["units_sold"] = p["units_sold"].interpolate(
    limit_direction="both"
)

p["lag_1"] = p["units_sold"].shift(1)
p["lag_2"] = p["units_sold"].shift(2)
p["lag_3"] = p["units_sold"].shift(3)

p["rolling_3"] = p["units_sold"].shift(1).rolling(3).mean()
p["month_num"] = p.index.month
p["time_idx"] = np.arange(len(p))

features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_3",
    "month_num",
    "time_idx",
]

model_data = p.dropna(
    subset=features + ["units_sold"]
).copy()

if len(model_data) < 10:
    st.warning(
        "Not enough monthly history for a meaningful demo forecast. "
        "Aim for at least 12 months of product history."
    )
else:
    split = min(
        max(6, int(len(model_data) * 0.8)),
        len(model_data) - 1,
    )

    train = model_data.iloc[:split]
    test = model_data.iloc[split:]

    eval_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
    )

    eval_model.fit(train[features], train["units_sold"])

    predicted = eval_model.predict(test[features])

    mae = mean_absolute_error(
        test["units_sold"],
        predicted,
    )

    rmse = mean_squared_error(
        test["units_sold"],
        predicted,
    ) ** 0.5

    f1, f2, f3 = st.columns(3)

    f1.metric("Holdout MAE", f"{mae:.1f} units")
    f2.metric("Holdout RMSE", f"{rmse:.1f} units")
    f3.metric("Training months", f"{len(train)}")

    eval_df = pd.DataFrame(
        {
            "Actual": test["units_sold"].values,
            "Predicted": predicted,
        },
        index=test.index,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=eval_df.index,
            y=eval_df.Actual,
            name="Actual",
            mode="lines+markers",
            line=dict(color="#111111", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=eval_df.index,
            y=eval_df.Predicted,
            name="Predicted",
            mode="lines+markers",
            line=dict(color="#B2D600", width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title="Chronological holdout · actual vs predicted",
        yaxis_title="Units sold",
        xaxis_title="Month",
    )

    st.plotly_chart(plot_layout(fig, 330), use_container_width=True)

    final_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
    )

    final_model.fit(
        model_data[features],
        model_data["units_sold"],
    )

    history = list(p["units_sold"].dropna().values)
    last_date = p.index.max()
    future = []

    for step in range(1, horizon + 1):
        future_date = last_date + pd.offsets.MonthBegin(step)

        row = {
            "lag_1": history[-1],
            "lag_2": history[-2] if len(history) > 1 else history[-1],
            "lag_3": history[-3] if len(history) > 2 else history[-1],
            "rolling_3": float(np.mean(history[-3:])),
            "month_num": future_date.month,
            "time_idx": len(p) + step - 1,
        }

        yhat = max(
            0,
            float(
                final_model.predict(
                    pd.DataFrame([row])[features]
                )[0]
            ),
        )

        future.append(
            {
                "Month": future_date,
                "Predicted units": int(round(yhat)),
            }
        )

        history.append(yhat)

    future_df = pd.DataFrame(future)

    fig = go.Figure()

    recent = p["units_sold"].tail(12)

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.values,
            name="Historical",
            mode="lines+markers",
            line=dict(color="#111111", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_df["Month"],
            y=future_df["Predicted units"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#B2D600", width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"{selected_product} · forecast",
        yaxis_title="Units sold",
        xaxis_title="Month",
    )

    st.plotly_chart(plot_layout(fig, 340), use_container_width=True)
    st.dataframe(future_df, use_container_width=True, hide_index=True)

    st.caption(
        "MAE and RMSE measure error on the chronological holdout. "
        "Forecast quality depends on data volume and quality; "
        "synthetic inputs produce demonstration results only."
    )


# ---------------------------------------------------------
# SECTION 4: BUSINESS ACTIONS
# ---------------------------------------------------------

st.markdown(
    "<div class='section-head'><span class='section-number'>04</span>"
    "<h2>Business actions</h2></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-note'>Rules-based insights derived from "
    "the loaded product dataset, plus practical next steps.</div>",
    unsafe_allow_html=True,
)

latest_month = monthly.month.max()

current = monthly[
    monthly.month == latest_month
][
    ["product_model", "units_sold", "revenue_usd"]
].copy()

prior_months = sorted(monthly.month.unique())

if len(prior_months) >= 2:
    previous = monthly[
        monthly.month == prior_months[-2]
    ][
        ["product_model", "units_sold"]
    ].rename(
        columns={"units_sold": "previous_units"}
    )

    current = current.merge(
        previous,
        on="product_model",
        how="left",
    )

    current["unit_change_pct"] = np.where(
        current.previous_units > 0,
        (current.units_sold / current.previous_units - 1) * 100,
        np.nan,
    )

    positive = current.dropna(
        subset=["unit_change_pct"]
    ).sort_values(
        "unit_change_pct",
        ascending=False,
    ).head(3)

    negative = current.dropna(
        subset=["unit_change_pct"]
    ).sort_values(
        "unit_change_pct"
    ).head(3)

    a, b = st.columns(2)

    with a:
        st.markdown("### Momentum to monitor")

        if positive.empty:
            st.info(
                "Not enough adjacent-month data to calculate momentum."
            )

        for _, row in positive.iterrows():
            st.markdown(
                f"<div class='insight-card' style='margin-bottom:.65rem'>"
                f"<div class='insight-label'>Month-on-month units</div>"
                f"<div class='insight-value'>↗ {row.product_model}</div>"
                f"<div class='insight-copy'>"
                f"{row.unit_change_pct:+.1f}% vs previous month. "
                "Check if the trend persists before increasing inventory."
                "</div></div>",
                unsafe_allow_html=True,
            )

    with b:
        st.markdown("### Declines to investigate")

        if negative.empty:
            st.info(
                "Not enough adjacent-month data to calculate declines."
            )

        for _, row in negative.iterrows():
            st.markdown(
                f"<div class='insight-card' style='margin-bottom:.65rem'>"
                f"<div class='insight-label'>Month-on-month units</div>"
                f"<div class='insight-value'>↘ {row.product_model}</div>"
                f"<div class='insight-copy'>"
                f"{row.unit_change_pct:+.1f}% vs previous month. "
                "Check stock availability, pricing, promotion, and seasonality."
                "</div></div>",
                unsafe_allow_html=True,
            )

st.markdown("### Recommended decision checklist")

st.markdown(
    """
    - **Inventory:** compare the forecast with on-hand stock and supplier
      lead times before replenishing.
    - **Discounts:** judge promotions using gross margin and returns—not
      units sold alone.
    - **Product performance:** investigate size-level stockouts, channel
      mix, customer reviews, and return reasons.
    - **Marketing:** compare campaign spend and conversion data before
      attributing a sales change to advertising.
    - **Data quality:** add region, channel, price, discount, returns,
      inventory, and marketing fields to improve future models.
    """
)

st.markdown(
    "<div class='note'><b>Interpretation guardrail</b> &nbsp; "
    "These are data-driven prompts, not causal conclusions. The demo "
    "dataset is synthetic and should not be used to make real inventory "
    "or investment decisions.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        EDUCATIONAL MLPA PROTOTYPE · Not an official Nike dashboard.
        Company financials are reported data; model-level sales and
        forecasts depend on the loaded dataset.
        <br>
        Source:
        <a href="https://investors.nike.com/sec-filings/all-sec-filings/content/0000320187-26-000088/nke-20260531.htm"
        target="_blank">Nike FY2026 Form 10-K</a>.
    </div>
    """,
    unsafe_allow_html=True,
)