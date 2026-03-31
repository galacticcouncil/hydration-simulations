import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta, timezone, date, time
import dateutil.parser

# =============================================================================
# PURE DATA FUNCTIONS
# =============================================================================

def get_kraken_prices(
        start_date: str | datetime,
        days: int | timedelta = 1,
        interval: timedelta | None = None,
        save_path: str | Path | None = None
) -> pd.DataFrame:
    import requests
    import time

    if isinstance(start_date, str):
        start_date = dateutil.parser.parse(start_date)

    start_ms = int(start_date.timestamp()) * 1000
    end_ms = int((start_date + (days if isinstance(days, timedelta) else timedelta(days=days))).timestamp()) * 1000

    start_str_formatted = start_date.astimezone(timezone.utc).strftime("%d %b, %Y %H:%M:%S")
    print(f"Fetching Kraken data starting from: {start_str_formatted}")

    interval_ms = int(interval.total_seconds()) * 1000 if interval else None
    current_interval = start_ms
    kraken_rows = []
    since = int(start_date.timestamp() * 1e9)

    INITIAL_RETRY_DELAY = 2
    MAX_RETRY_DELAY = 60
    MAX_RETRIES = 8

    while True:
        retry_delay = INITIAL_RETRY_DELAY
        for attempt in range(MAX_RETRIES):
            resp = requests.get(
                "https://api.kraken.com/0/public/Trades",
                params={"pair": "EURUSD", "since": since},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("error"):
                break

            if any("Too many requests" in e for e in data["error"]):
                print(f"Rate limited, retrying in {retry_delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            else:
                raise RuntimeError(f"Kraken API error: {data['error']}")
        else:
            raise RuntimeError(f"Kraken API rate limit not resolved after {MAX_RETRIES} retries")

        result = data["result"]
        pair_key = next(k for k in result if k != "last")
        trades = result[pair_key]
        since = int(result["last"])

        if not trades:
            break

        done = False
        for trade in trades:
            ts_ms = int(float(trade[2]) * 1000)

            if ts_ms > end_ms:
                done = True
                break
            if interval_ms:
                if ts_ms < current_interval:
                    continue
                current_interval += interval_ms

            kraken_rows.append({
                "timestamp_ms": ts_ms,
                "readable_time": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                "pair": "EURUSD",
                "price": float(trade[0])
            })

        if done:
            break

        last_ts_ms = int(float(trades[-1][2]) * 1000)
        if last_ts_ms >= end_ms:
            break

    df_kraken = pd.DataFrame(kraken_rows)
    if save_path:
        output_file = Path(save_path)
        df_kraken.to_csv(output_file, index=False)

    return df_kraken


def get_binance_prices(
        start_date: str or datetime,
        days: int or timedelta = 1,
        interval: timedelta or None = None,
        save_path: str or Path or None = None
) -> pd.DataFrame:
    from binance.client import Client
    client = Client()

    if isinstance(start_date, str):
        start_date = dateutil.parser.parse(start_date)
    start_ms = int(start_date.timestamp()) * 1000
    end_ms = int((start_date + (days if isinstance(days, timedelta) else timedelta(days=days))).timestamp()) * 1000
    # convert to utc time
    start_str_formatted = start_date.astimezone(timezone.utc).strftime("%d %b, %Y %H:%M:%S")

    print(f"Fetching Binance data starting from: {start_str_formatted}")

    agg_trades = client.aggregate_trade_iter(
        symbol='EURUSDT',
        start_str=start_str_formatted
    )

    interval_ms = int(interval.total_seconds()) * 1000 if interval else None
    current_interval = start_ms
    binance_rows = []
    for trade in agg_trades:
        ts = int(trade['T'])

        # Only include data points that fall within the Kraken range
        if ts >= start_ms:
            if interval_ms:
                # If an interval is specified, only include trades at the specified intervals
                if ts < current_interval:
                    continue
                current_interval += interval_ms
            # Map to your specific headers: timestamp_ms, readable_time, pair, price
            binance_rows.append({
                "timestamp_ms": ts,
                "readable_time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[
                                 :-3],
                "pair": "EURUSD",  # Standardizing to match your Kraken 'pair' column
                "price": float(trade['p'])
            })

        if ts > end_ms:
            break

    df_binance = pd.DataFrame(binance_rows)
    if save_path:
        output_file = Path(save_path)
        df_binance.to_csv(output_file, index=False)

    return df_binance


def get_prices_for_day(exchange_name: str, day: date) -> pd.DataFrame:
    cache_dir = Path(__file__).parent / 'price data' / exchange_name
    cache_file = cache_dir / f'{day.strftime("%Y-%m-%d")}.csv'

    if cache_file.exists():
        print(f"Loading cached {exchange_name} data for {day}")
        return pd.read_csv(cache_file)

    cache_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.combine(day, time.min, tzinfo=timezone.utc)

    fetchers = {
        'binance': get_binance_prices,
        'kraken': get_kraken_prices,
    }
    if exchange_name not in fetchers:
        raise ValueError(f"Unknown exchange '{exchange_name}'. Expected one of: {list(fetchers)}")

    return fetchers[exchange_name](start_date=start_dt, days=1, save_path=cache_file)


def detect_and_load(f, local_tz):
    """
    Accepts a file path (str) or a file-like object.
    Detects format from column headers and returns a normalized DataFrame
    with just 'time' (UTC-aware) and 'price' columns.
    """
    df = pd.read_csv(f)
    cols = set(df.columns)

    if "timestamp_ms" in cols:
        # Format A: timestamp_ms, readable_time, pair, price
        df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    elif "timestamp" in cols:
        # Format B: timestamp, pair, EUR/USD
        price_col = next((c for c in df.columns if c not in ("timestamp", "pair")), None)
        if price_col is None:
            raise ValueError("Could not find price column in file.")
        df["price"] = pd.to_numeric(
            df[price_col].astype(str).str.replace("$", "", regex=False),
            errors="coerce"
        )
        df["time"] = (
            pd.to_datetime(df["timestamp"], errors="coerce")
              .dt.tz_localize(local_tz)
              .dt.tz_convert("UTC")
        )

    else:
        raise ValueError(
            f"Unrecognized columns: {list(df.columns)}. "
            "Expected 'timestamp_ms' or 'timestamp' as the time column."
        )

    return df[["time", "price"]].dropna().sort_values("time")


def build_merged(df1, df2):
    """
    Combine two (time, price) DataFrames so that every exact data point from
    either source is paired with the most recent actual reading from the other
    source (forward-fill, no interpolation). Only rows where both sources have
    been seen at least once are kept, trimmed to the overlapping time window.
    """
    a = df1[["time", "price"]].copy(); a["src"] = 1
    b = df2[["time", "price"]].copy(); b["src"] = 2
    combined = pd.concat([a, b]).sort_values("time").reset_index(drop=True)

    combined["price1"] = combined["price"].where(combined["src"] == 1).ffill()
    combined["price2"] = combined["price"].where(combined["src"] == 2).ffill()

    overlap_start = max(df1["time"].min(), df2["time"].min())
    overlap_end   = min(df1["time"].max(), df2["time"].max())

    merged = (
        combined[["time", "price1", "price2"]]
        .dropna()
        .query("@overlap_start <= time <= @overlap_end")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return merged, overlap_start, overlap_end


def zero_crossing_segments(times, values):
    """
    Split a series into continuous same-sign segments.
    At each sign flip, insert an interpolated zero-crossing point so
    both adjacent segments share that endpoint — no gaps, clean crossing.
    Returns list of (times, values, is_positive) tuples.
    """
    segments = []
    seg_t = [times[0]]
    seg_v = [values[0]]

    for i in range(1, len(times)):
        v_prev, v_curr = values[i - 1], values[i]
        if v_prev * v_curr < 0:  # sign change
            t0 = pd.Timestamp(times[i - 1]).timestamp()
            t1 = pd.Timestamp(times[i]).timestamp()
            frac = v_prev / (v_prev - v_curr)  # fraction of interval where line = 0
            t_zero = pd.Timestamp(t0 + frac * (t1 - t0), unit="s", tz="UTC")
            seg_t.append(t_zero); seg_v.append(0.0)
            segments.append((seg_t, seg_v, v_prev >= 0))
            seg_t = [t_zero, times[i]]
            seg_v = [0.0, v_curr]
        else:
            seg_t.append(times[i])
            seg_v.append(v_curr)

    segments.append((seg_t, seg_v, seg_v[0] >= 0))
    return segments


def decimate_spread(times, values, n_buckets):
    """
    Divide the time range into n_buckets equal-width buckets. For each bucket,
    keep the timestamp/value of the max positive spread (if any) and the max
    negative spread (if any). Results are sorted by time so the line draws
    correctly. This caps rendered points at roughly 2 * n_buckets regardless
    of how dense the underlying data is.
    """
    if not times:
        return [], []

    t0 = pd.Timestamp(times[0]).timestamp()
    t1 = pd.Timestamp(times[-1]).timestamp()
    span = t1 - t0
    if span == 0:
        return list(times), list(values)

    bucket_secs = span / n_buckets

    # For each bucket track the best positive and best negative point
    pos_best = {}  # bucket_idx -> (time, value)
    neg_best = {}

    for t, v in zip(times, values):
        idx = min(int((pd.Timestamp(t).timestamp() - t0) / bucket_secs), n_buckets - 1)
        if v >= 0:
            if idx not in pos_best or v > pos_best[idx][1]:
                pos_best[idx] = (t, v)
        else:
            if idx not in neg_best or v < neg_best[idx][1]:
                neg_best[idx] = (t, v)

    # Merge and sort by time
    all_points = list(pos_best.values()) + list(neg_best.values())
    all_points.sort(key=lambda p: pd.Timestamp(p[0]))

    keep_t = [p[0] for p in all_points]
    keep_v = [p[1] for p in all_points]
    return keep_t, keep_v

def run_comparison(file1_path, file2_path, local_tz="America/Chicago", n_buckets=400):
    """
    Load, parse, merge, and compute spread for two EUR/USD CSV files.
    Returns a dict with:
      df1, df2        — normalized per-source DataFrames (time, price)
      merged          — combined DataFrame (time, price1, price2, spread_pct)
      hover_t/hover_v — decimated time/value lists used for the spread graph
      stats           — dict of summary statistics
    Raises ValueError with a descriptive message on any parsing failure.
    """
    df1 = detect_and_load(file1_path, local_tz)
    df2 = detect_and_load(file2_path, local_tz)

    merged, overlap_start, overlap_end = build_merged(df1, df2)

    if merged.empty:
        raise ValueError(
            f"No overlapping time range between files.\n"
            f"  File 1: {df1['time'].min()} -> {df1['time'].max()}\n"
            f"  File 2: {df2['time'].min()} -> {df2['time'].max()}\n"
            f"  Check the local_tz setting (currently '{local_tz}')."
        )

    merged["spread_pct"] = ((merged["price1"] - merged["price2"]) / merged["price2"] * 100).round(4)

    h_t, h_v = decimate_spread(
        merged["time"].tolist(),
        merged["spread_pct"].tolist(),
        n_buckets,
    )

    max_pos_row = merged.loc[merged["spread_pct"].idxmax()]
    max_neg_row = merged.loc[merged["spread_pct"].idxmin()]

    stats = {
        "n_points":        len(merged),
        "n_hover_points":  len(h_t),
        "overlap_start":   overlap_start,
        "overlap_end":     overlap_end,
        "max_spread":      max_pos_row["spread_pct"],
        "max_spread_time": max_pos_row["time"],
        "min_spread":      max_neg_row["spread_pct"],
        "min_spread_time": max_neg_row["time"],
        "mean_spread":     round(merged["spread_pct"].mean(), 4),
        "std_spread":      round(merged["spread_pct"].std(), 4),
    }

    return {
        "df1":     df1,
        "df2":     df2,
        "merged":  merged,
        "hover_t": h_t,
        "hover_v": h_v,
        "stats":   stats,
    }

def _to_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", "price"]).sort_values("timestamp_ms")
    df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df[["timestamp_ms", "time", "price"]]


def _interp_prices(target_ms: np.ndarray, source_ms: np.ndarray, source_prices: np.ndarray) -> np.ndarray:
    if len(source_ms) == 0:
        return np.full_like(target_ms, np.nan, dtype=float)
    if len(source_ms) == 1:
        return np.full_like(target_ms, float(source_prices[0]), dtype=float)

    interp = np.interp(target_ms, source_ms, source_prices).astype(float)
    out_of_range = (target_ms < source_ms.min()) | (target_ms > source_ms.max())
    interp[out_of_range] = np.nan
    return interp


def smooth_binance_with_kraken(binance_df: pd.DataFrame, kraken_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Binance timestamp, interpolate the Kraken price and choose
    whichever (Binance or interpolated Kraken) is closer to the last
    chosen price. Returns a DataFrame keyed on Binance timestamps.
    """
    binance = _to_price_df(binance_df)
    kraken = _to_price_df(kraken_df)

    if binance.empty:
        return pd.DataFrame(columns=[
            "timestamp_ms", "time", "binance_price", "kraken_price", "smoothed_price"
        ])

    target_ms = binance["timestamp_ms"].to_numpy(dtype=np.int64)
    b_prices = binance["price"].to_numpy(dtype=float)

    k_times = kraken["timestamp_ms"].to_numpy(dtype=np.int64)
    k_prices = kraken["price"].to_numpy(dtype=float)

    interp = _interp_prices(target_ms, k_times, k_prices)

    smoothed = []
    last_value = float(b_prices[0])
    for b_price, k_price in zip(b_prices, interp):
        if np.isnan(k_price):
            chosen = b_price
        else:
            if abs(b_price - last_value) <= abs(k_price - last_value):
                chosen = b_price
            else:
                chosen = k_price
        smoothed.append(chosen)
        last_value = chosen

    out = pd.DataFrame({
        "timestamp_ms": binance["timestamp_ms"],
        "time": binance["time"],
        "binance_price": b_prices,
        "kraken_price": interp,
        "smoothed_price": np.array(smoothed, dtype=float),
    })
    return out


# =============================================================================
# IMPORTABLE DEBUG ENTRY POINT
# Usage:
#   from price_compare import run_comparison
#   result = run_comparison("file_a.csv", "file_b.csv", local_tz="America/Chicago")
#   print(result["stats"])
#   print(result["merged"].head(20))
# Also runnable directly:
#   python price_compare.py file_a.csv file_b.csv [timezone]
# =============================================================================


if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) >= 3:
        tz = sys.argv[3] if len(sys.argv) > 3 else "America/Chicago"
        result = run_comparison(sys.argv[1], sys.argv[2], local_tz=tz)
        pprint.pprint(result["stats"])
        print()
        print(result["merged"].head(20).to_string())
    else:
        print("Usage: python price_compare.py <file1.csv> <file2.csv> [timezone]")


# =============================================================================
# STREAMLIT UI  (only runs when launched via `streamlit run price_compare.py`)
# =============================================================================
if st.runtime.exists():
    st.set_page_config(page_title="EUR/USD Price Comparison", layout="wide")
    st.title("EUR/USD Price Source Comparison")

    with st.sidebar:
        st.header("Settings")
        date_range = st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=1), date.today() - timedelta(days=1)),
            max_value=date.today() - timedelta(days=1),
        )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_day, end_day = date_range
    else:
        st.info("Select a start and end date to begin.")
        st.stop()

    days_in_range = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]

    if st.button("Fetch Data", type="primary", key="fetch_binance_kraken"):
        st.session_state.pop("binance_df", None)
        st.session_state.pop("kraken_df", None)

        with st.spinner("Fetching Binance data..."):
            try:
                binance_df = pd.concat(
                    [get_prices_for_day("binance", d) for d in days_in_range],
                    ignore_index=True
                )
                st.session_state["binance_df"] = binance_df
            except Exception as exc:
                st.error(f"Failed to fetch Binance data: {exc}")
                st.stop()

        with st.spinner("Fetching Kraken data..."):
            try:
                kraken_df = pd.concat(
                    [get_prices_for_day("kraken", d) for d in days_in_range],
                    ignore_index=True
                )
                st.session_state["kraken_df"] = kraken_df
            except Exception as exc:
                st.error(f"Failed to fetch Kraken data: {exc}")
                st.stop()

    if "binance_df" not in st.session_state or "kraken_df" not in st.session_state:
        st.info("Select a date range, then click **Fetch Data**.")
        st.stop()

    smoothed_df = smooth_binance_with_kraken(
        st.session_state["binance_df"],
        st.session_state["kraken_df"],
    )

    if smoothed_df.empty:
        st.warning("No data available for the selected range.")
        st.stop()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=smoothed_df["time"], y=smoothed_df["binance_price"],
            name="Binance",
            line=dict(color="#4C9BE8", width=1.2, shape="hv"),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Binance</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=smoothed_df["time"], y=smoothed_df["kraken_price"],
            name="Kraken (interp)",
            line=dict(color="#F4A83A", width=1.2, dash="dot", shape="hv"),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Kraken</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=smoothed_df["time"], y=smoothed_df["smoothed_price"],
            name="Combined",
            line=dict(color="#2ECC71", width=1.6),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Combined</extra>",
        )
    )

    fig.update_layout(
        height=650,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.update_yaxes(title_text="EUR/USD", tickformat=".5f")
    fig.update_xaxes(title_text="Time (UTC)")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View combined data table"):
        display = smoothed_df[["time", "binance_price", "kraken_price", "smoothed_price"]].copy()
        display["time"] = display["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        for col in ["binance_price", "kraken_price", "smoothed_price"]:
            display[col] = display[col].map(lambda v: f"{v:.6f}" if pd.notna(v) else "")
        display.columns = ["Time (UTC)", "Binance", "Kraken (interp)", "Combined"]
        st.dataframe(display, use_container_width=True, hide_index=True)
