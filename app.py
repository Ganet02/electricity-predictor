"""
Phase 3: Interactive Web Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Electricity Peak-Demand Predictor", layout="wide")
st.title("⚡ Electricity Peak-Demand Predictor & Bill Estimator")

# ============================================================
# File upload (defaults to the sample dataset if nothing is uploaded)
# ============================================================
DEFAULT_FILE = "customer_usage_week.csv"

uploaded_file = st.file_uploader(
    "Upload a different customer usage CSV (optional - sample data is shown by default)",
    type="csv"
)

if uploaded_file is not None:
    data_source = uploaded_file
    st.caption("Showing results for your uploaded file.")
else:
    data_source = DEFAULT_FILE
    st.caption("Showing results for the sample dataset (CUST-102). Upload your own CSV above to replace it.")


# ============================================================
# Clean data (same logic as Phase 2)
# ============================================================
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


df = load_and_clean(data_source)
st.success(f"Loaded {len(df)} readings for {df['Customer_ID'].iloc[0]}")


# ============================================================
# Bill calculator
# ============================================================
def bill_calculator(usage):
    if usage <= 500:
        return usage * 0.12
    return (500 * 0.12) + ((usage - 500) * 0.18)


total_kwh = df["kWh_Used"].sum()
bill = bill_calculator(total_kwh)
peak_row = df.loc[df["kWh_Used"].idxmax()]

col1, col2, col3 = st.columns(3)
col1.metric("Total Usage", f"{round(total_kwh, 2)} kWh")
col2.metric("Estimated Bill", f"${round(bill, 2)}")
col3.metric("Peak Usage", f"{peak_row['kWh_Used']} kWh", f"at {peak_row['Timestamp']}")


# ============================================================
# Usage chart
# ============================================================
st.subheader("Daily Usage Trend")
daily = df.groupby("Date")["kWh_Used"].sum()
fig, ax = plt.subplots(figsize=(10, 4))
daily.plot(kind="line", marker="o", ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("Total kWh")
ax.grid(True)
st.pyplot(fig)


# ============================================================
# Predictive model
# ============================================================
st.subheader("Next Week's Predicted Usage")

features = ["Hour", "DayOfWeek", "IsWeekend", "Temperature (°C)"]
X = df[features]
y = df["kWh_Used"]

if len(df) < 20:
    st.warning("Not enough data to train a reliable model (need at least ~20+ readings).")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    st.caption(f"Model accuracy — Mean Absolute Error: {round(mae, 2)} kWh | R²: {round(r2, 2)} "
               f"(low R² means limited historical data; predictions are a rough estimate)")

    last_date = df["Timestamp"].max()
    future_timestamps = pd.date_range(last_date + pd.Timedelta(hours=1), periods=168, freq="h")
    avg_temp_by_hour = df.groupby("Hour")["Temperature (°C)"].mean()

    future_df = pd.DataFrame({"Timestamp": future_timestamps})
    future_df["Hour"] = future_df["Timestamp"].dt.hour
    future_df["DayOfWeek"] = future_df["Timestamp"].dt.dayofweek
    future_df["IsWeekend"] = (future_df["DayOfWeek"] >= 5).astype(int)
    future_df["Temperature (°C)"] = future_df["Hour"].map(avg_temp_by_hour)
    future_df["Predicted_kWh"] = model.predict(future_df[features])

    predicted_total = future_df["Predicted_kWh"].sum()
    predicted_bill = bill_calculator(predicted_total)

    col1, col2 = st.columns(2)
    col1.metric("Predicted Usage (next 7 days)", f"{round(predicted_total, 2)} kWh")
    col2.metric("Predicted Bill", f"${round(predicted_bill, 2)}")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    future_daily = future_df.groupby(future_df["Timestamp"].dt.date)["Predicted_kWh"].sum()
    future_daily.plot(kind="bar", ax=ax2, color="orange")
    ax2.set_ylabel("Predicted kWh")
    ax2.set_xlabel("Date")
    st.pyplot(fig2)

    # ============================================================
    # Time-of-use optimizer
    # ============================================================
    st.subheader("💡 Time-of-Use Savings Tips")
    PEAK_HOURS = list(range(17, 21))
    OFFPEAK_HOURS = list(range(0, 6))
    BASE_RATE = 0.12
    PEAK_RATE = BASE_RATE * 1.5

    offpeak_avg = future_df[future_df["Hour"].isin(OFFPEAK_HOURS)].groupby("Hour")["Predicted_kWh"].mean()
    if not offpeak_avg.empty:
        cheapest_hour = offpeak_avg.idxmin()
        flexible_load = 2.0
        savings = flexible_load * (PEAK_RATE - BASE_RATE)
        st.write(
            f"Your predicted peak hours are **{PEAK_HOURS[0]}:00–{PEAK_HOURS[-1]}:00**. "
            f"Shifting a flexible appliance (e.g. washing machine, ~{flexible_load} kWh) to "
            f"**{cheapest_hour}:00** instead could save about **${round(savings, 2)} per use** "
            f"(~${round(savings * 4, 2)}/month if done weekly)."
        )

# ============================================================
# Raw data viewer
# ============================================================
with st.expander("View raw data"):
    st.dataframe(df)
