"""
Phase 3: Interactive Web Dashboard (v2 - interactive Plotly charts)
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Electricity Peak-Demand Predictor",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------
# Look and feel
# ------------------------------------------------------------
AMBER = "#FFB703"   # actual usage / peak
TEAL = "#2EC4B6"    # predicted usage
SLATE = "#8FA3BF"   # neutral / reference lines

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #16233A;
        border-left: 4px solid #FFB703;
        border-radius: 6px;
        padding: 14px 16px;
    }
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, height=380):
    """Apply one consistent look to every Plotly chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ------------------------------------------------------------
# Sidebar: data + tariff settings
# ------------------------------------------------------------
DEFAULT_FILE = "customer_usage_week.csv"
FEATURES = ["Hour", "DayOfWeek", "IsWeekend", "Temperature (°C)"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader(
        "Upload a customer usage CSV (optional)", type="csv",
        help="Leave empty to use the sample dataset.",
    )

    st.header("Tariff")
    currency = st.text_input("Currency symbol", value="$", max_chars=5)
    base_rate = st.number_input("Base rate (per kWh)", min_value=0.0, value=0.12, step=0.01)
    tier_limit = st.number_input("Tier limit (kWh)", min_value=1, value=500, step=50)
    high_rate = st.number_input("Rate above tier limit (per kWh)", min_value=0.0, value=0.18, step=0.01)

    st.header("Time-of-use")
    peak_start, peak_end = st.slider(
        "Peak window (hours of the day)", 0, 24, (17, 21),
        help="Peak runs from the first hour up to, but not including, the second.",
    )
    peak_multiplier = st.slider("Peak rate multiplier", 1.0, 3.0, 1.5, 0.1)

    st.header("Forecast")
    horizon_days = st.slider("Days to forecast", 3, 14, 7)

PEAK_HOURS = list(range(peak_start, peak_end))
OFFPEAK_HOURS = list(range(0, 6))
PEAK_RATE = base_rate * peak_multiplier

st.title("⚡ Electricity Peak-Demand Predictor & Bill Estimator")


# ------------------------------------------------------------
# Data loading and cleaning (same logic as Phase 2)
# ------------------------------------------------------------
@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.drop_duplicates()
    temp_col = "Temperature (°C)"
    df[temp_col] = df[temp_col].fillna(df[temp_col].mean())
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Date"] = df["Timestamp"].dt.date
    return df.sort_values("Timestamp").reset_index(drop=True)


data_source = uploaded_file if uploaded_file is not None else DEFAULT_FILE
try:
    df = load_and_clean(data_source)
    missing = {"Timestamp", "kWh_Used", "Temperature (°C)", "Customer_ID"} - set(df.columns)
    if missing:
        raise KeyError(", ".join(sorted(missing)))
except Exception as e:
    st.error(
        "Could not read the data. The CSV needs these columns: Timestamp, "
        f"Customer_ID, kWh_Used, Temperature (°C). Details: {e}"
    )
    st.stop()

source_label = "your uploaded file" if uploaded_file is not None else "the sample dataset"
st.caption(f"Showing {len(df)} readings for **{df['Customer_ID'].iloc[0]}** from {source_label}.")


# ------------------------------------------------------------
# Bill calculator (tier thresholds and rates come from the sidebar)
# ------------------------------------------------------------
def bill_calculator(usage):
    if usage <= tier_limit:
        return usage * base_rate
    return (tier_limit * base_rate) + ((usage - tier_limit) * high_rate)


def money(x):
    return f"{currency}{x:,.2f}"


# ------------------------------------------------------------
# Model (trained once, cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Training the model...")
def train_models(data):
    X, y = data[FEATURES], data["kWh_Used"]
    split = int(len(data) * 0.8)  # chronological split: test on the most recent 20%
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    eval_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    pred = eval_model.predict(X_test)

    final_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    final_model.fit(X, y)  # refit on all data for the forecast

    return {
        "final_model": final_model,
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "baseline_mae": mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())),
        "test_ts": data["Timestamp"].iloc[split:],
        "y_test": y_test,
        "pred": pred,
        "importances": pd.Series(final_model.feature_importances_, index=FEATURES),
    }


tab_overview, tab_forecast, tab_savings, tab_model, tab_data = st.tabs(
    ["Overview", "Forecast", "Savings", "Model", "Data"]
)

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    total_kwh = df["kWh_Used"].sum()
    bill = bill_calculator(total_kwh)
    peak_row = df.loc[df["kWh_Used"].idxmax()]
    n_days = max(df["Date"].nunique(), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total usage", f"{total_kwh:,.1f} kWh")
    c2.metric("Estimated bill", money(bill))
    c3.metric(
        "Highest reading", f"{peak_row['kWh_Used']} kWh",
        f"{peak_row['Timestamp']:%a %d %b, %H:%M}", delta_color="off",
    )
    c4.metric("Average per day", f"{total_kwh / n_days:,.1f} kWh")

    left, right = st.columns([3, 2])

    with left:
        daily = df.groupby("Date")["kWh_Used"].sum().reset_index()
        fig = px.line(daily, x="Date", y="kWh_Used", markers=True,
                      title="Daily usage", labels={"kWh_Used": "kWh"})
        fig.update_traces(line_color=AMBER, hovertemplate="%{x}<br>%{y:.1f} kWh<extra></extra>")
        st.plotly_chart(style(fig), key="daily_trend")

    with right:
        tier1 = min(total_kwh, tier_limit) * base_rate
        tier2 = max(total_kwh - tier_limit, 0) * high_rate
        parts = pd.DataFrame({
            "Part": [f"Up to {tier_limit} kWh", f"Above {tier_limit} kWh"],
            "Cost": [tier1, tier2],
        })
        parts = parts[parts["Cost"] > 0]
        fig = px.pie(parts, names="Part", values="Cost", hole=0.55,
                     title="Where the bill comes from",
                     color_discrete_sequence=[AMBER, TEAL])
        fig.update_traces(hovertemplate="%{label}<br>" + currency + "%{value:,.2f}<extra></extra>")
        st.plotly_chart(style(fig), key="bill_split")

    fig = px.line(df, x="Timestamp", y="kWh_Used", title="Every reading (drag the slider to zoom)",
                  labels={"kWh_Used": "kWh"})
    fig.update_traces(line_color=AMBER)
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(style(fig, 420), key="all_readings")

    left, right = st.columns(2)

    with left:
        profile = df.assign(Day=np.where(df["IsWeekend"] == 1, "Weekend", "Weekday"))
        profile = profile.groupby(["Hour", "Day"])["kWh_Used"].mean().reset_index()
        fig = px.line(profile, x="Hour", y="kWh_Used", color="Day", markers=True,
                      title="Typical day: weekday vs weekend",
                      labels={"kWh_Used": "Average kWh"},
                      color_discrete_map={"Weekday": AMBER, "Weekend": TEAL})
        if PEAK_HOURS:
            fig.add_vrect(x0=PEAK_HOURS[0] - 0.5, x1=PEAK_HOURS[-1] + 0.5,
                          fillcolor=SLATE, opacity=0.2, line_width=0,
                          annotation_text="Peak window", annotation_position="top left")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig), key="typical_day")

    with right:
        heat = df.pivot_table(index="Hour", columns="DayOfWeek", values="kWh_Used", aggfunc="mean")
        heat = heat.reindex(columns=range(7))
        heat.columns = DAY_NAMES
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                        title="When demand is highest (kWh)", origin="lower",
                        labels=dict(x="Day", y="Hour", color="kWh"))
        st.plotly_chart(style(fig), key="heatmap")

# ============================================================
# Everything below needs a trained model
# ============================================================
enough_data = len(df) >= 20

if enough_data:
    results = train_models(df)
    model = results["final_model"]

    last_ts = df["Timestamp"].max()
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=24 * horizon_days, freq="h")
    avg_temp_by_hour = df.groupby("Hour")["Temperature (°C)"].mean()

    future_df = pd.DataFrame({"Timestamp": future_ts})
    future_df["Hour"] = future_df["Timestamp"].dt.hour
    future_df["DayOfWeek"] = future_df["Timestamp"].dt.dayofweek
    future_df["IsWeekend"] = (future_df["DayOfWeek"] >= 5).astype(int)
    future_df["Temperature (°C)"] = (
        future_df["Hour"].map(avg_temp_by_hour).fillna(df["Temperature (°C)"].mean())
    )
    future_df["Predicted_kWh"] = model.predict(future_df[FEATURES])

    # spread of the individual trees = a rough uncertainty band
    tree_preds = np.stack([t.predict(future_df[FEATURES].values) for t in model.estimators_])
    future_df["Low"] = np.percentile(tree_preds, 10, axis=0)
    future_df["High"] = np.percentile(tree_preds, 90, axis=0)
    future_df["Date"] = future_df["Timestamp"].dt.date

    predicted_total = future_df["Predicted_kWh"].sum()
    predicted_bill = bill_calculator(predicted_total)

# ============================================================
# FORECAST
# ============================================================
with tab_forecast:
    if not enough_data:
        st.warning("Not enough data to train a reliable model (need at least ~20 readings).")
    else:
        top = future_df.loc[future_df["Predicted_kWh"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Predicted usage ({horizon_days} days)", f"{predicted_total:,.1f} kWh")
        c2.metric("Predicted bill", money(predicted_bill))
        c3.metric("Predicted peak hour", f"{top['Timestamp']:%a %H:%M}",
                  f"{top['Predicted_kWh']:.2f} kWh", delta_color="off")

        fig = go.Figure()
        fig.add_scatter(x=future_df["Timestamp"], y=future_df["High"], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=future_df["Timestamp"], y=future_df["Low"], mode="lines",
                        line=dict(width=0), fill="tonexty", fillcolor="rgba(46,196,182,0.25)",
                        name="Likely range (10th–90th percentile of trees)", hoverinfo="skip")
        fig.add_scatter(x=future_df["Timestamp"], y=future_df["Predicted_kWh"], mode="lines",
                        line=dict(color=TEAL), name="Predicted kWh",
                        hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>")
        fig.update_layout(title="Hour-by-hour forecast", yaxis_title="kWh", hovermode="x unified")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(style(fig, 440), key="forecast_hourly")

        daily_f = future_df.groupby("Date")["Predicted_kWh"].sum().reset_index()
        fig = px.bar(daily_f, x="Date", y="Predicted_kWh", title="Predicted usage per day",
                     labels={"Predicted_kWh": "Predicted kWh"})
        fig.update_traces(marker_color=TEAL,
                          hovertemplate="%{x}<br>%{y:.1f} kWh<extra></extra>")
        st.plotly_chart(style(fig), key="forecast_daily")

        st.caption("The shaded band shows how much the model's trees disagree. "
                   "It is a rough guide to uncertainty, not a guaranteed range.")

# ============================================================
# SAVINGS
# ============================================================
with tab_savings:
    if not enough_data:
        st.warning("Not enough data to train a reliable model (need at least ~20 readings).")
    elif not PEAK_HOURS:
        st.info("Set a peak window of at least one hour in the sidebar.")
    else:
        st.subheader("💡 Time-of-use savings")

        by_hour = future_df.groupby("Hour")["Predicted_kWh"].mean().reset_index()
        by_hour["Period"] = np.where(by_hour["Hour"].isin(PEAK_HOURS), "Peak", "Other")
        fig = px.bar(by_hour, x="Hour", y="Predicted_kWh", color="Period",
                     title="Predicted average usage by hour",
                     labels={"Predicted_kWh": "Average kWh"},
                     color_discrete_map={"Peak": AMBER, "Other": SLATE})
        st.plotly_chart(style(fig), key="savings_by_hour")

        offpeak_avg = (future_df[future_df["Hour"].isin(OFFPEAK_HOURS)]
                       .groupby("Hour")["Predicted_kWh"].mean())
        if not offpeak_avg.empty:
            cheapest_hour = offpeak_avg.idxmin()
            flexible_load = 2.0
            per_use = flexible_load * (PEAK_RATE - base_rate)
            st.write(
                f"Your predicted peak hours are **{PEAK_HOURS[0]}:00–{PEAK_HOURS[-1] + 1}:00**. "
                f"Shifting a flexible appliance (e.g. a washing machine, about {flexible_load} kWh) to "
                f"**{cheapest_hour}:00** could save about **{money(per_use)} per use** "
                f"(around {money(per_use * 4)} a month if done weekly)."
            )

        st.divider()
        st.subheader("What if you shift some peak usage?")
        shift_pct = st.slider("Share of peak-hour usage moved to off-peak", 0, 100, 20, 5, format="%d%%")
        peak_kwh = future_df.loc[future_df["Hour"].isin(PEAK_HOURS), "Predicted_kWh"].sum()
        shifted_kwh = peak_kwh * shift_pct / 100
        saving = shifted_kwh * (PEAK_RATE - base_rate)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted peak-hour usage", f"{peak_kwh:,.1f} kWh")
        c2.metric("Usage moved", f"{shifted_kwh:,.1f} kWh")
        c3.metric(f"Saved over {horizon_days} days", money(saving))

# ============================================================
# MODEL
# ============================================================
with tab_model:
    if not enough_data:
        st.warning("Not enough data to train a reliable model (need at least ~20 readings).")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean absolute error", f"{results['mae']:.2f} kWh")
        c2.metric("R²", f"{results['r2']:.2f}")
        c3.metric("Error of a 'just use the average' guess", f"{results['baseline_mae']:.2f} kWh")
        st.caption(
            "Tested on the most recent 20% of readings, which the model had not seen. "
            "If the model's error is not clearly below the 'average' guess, it has learned little "
            "yet. More history will help."
        )

        test_df = pd.DataFrame({
            "Timestamp": results["test_ts"].values,
            "Actual": results["y_test"].values,
            "Predicted": results["pred"],
        })
        fig = px.line(test_df, x="Timestamp", y=["Actual", "Predicted"],
                      title="Actual vs predicted on unseen data",
                      color_discrete_map={"Actual": AMBER, "Predicted": TEAL},
                      labels={"value": "kWh", "variable": ""})
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig), key="actual_vs_pred")

        left, right = st.columns(2)
        with left:
            imp = results["importances"].sort_values().reset_index()
            imp.columns = ["Feature", "Importance"]
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title="What drives the prediction")
            fig.update_traces(marker_color=AMBER)
            st.plotly_chart(style(fig), key="importance")
        with right:
            lim = max(test_df["Actual"].max(), test_df["Predicted"].max())
            fig = px.scatter(test_df, x="Actual", y="Predicted",
                             title="Closer to the line is better")
            fig.update_traces(marker=dict(color=TEAL, size=8))
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                          line=dict(color=SLATE, dash="dash"))
            st.plotly_chart(style(fig), key="scatter")

# ============================================================
# DATA
# ============================================================
with tab_data:
    st.dataframe(df, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download cleaned data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_usage.csv", mime="text/csv")
    if enough_data:
        out = future_df[["Timestamp", "Predicted_kWh", "Low", "High"]]
        c2.download_button("Download forecast (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="forecast.csv", mime="text/csv")
