import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from datetime import datetime, timedelta, timezone, date, time
import dateutil.parser

# =============================================================================
# PURE DATA FUNCTIONS  (no Streamlit — safe to import and test independently)
# =============================================================================

def get_kraken_prices(
        start_date: str | datetime,
        days: int | timedelta = 1,
        interval: timedelta | None = None,
        save_path: str | Path | None = None
) -> pd.DataFrame:
    import requests

    if isinstance(start_date, str):
        start_date = dateutil.parser.parse(start_date)

    start_ns = int(start_date.timestamp() * 1e9)  # Kraken 'since' uses nanoseconds
    end_ms = int((start_date + (days if isinstance(days, timedelta) else timedelta(days=days))).timestamp()) * 1000

    start_str_formatted = start_date.astimezone(timezone.utc).strftime("%d %b, %Y %H:%M:%S")
    print(f"Fetching Kraken data starting from: {start_str_formatted}")

    interval_ms = int(interval.total_seconds()) * 1000 if interval else None
    current_interval = int(start_date.timestamp()) * 1000
    kraken_rows = []
    since = start_ns

    while True:
        resp = requests.get(
            "https://api.kraken.com/0/public/Trades",
            params={"pair": "EURUSD", "since": since},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")

        result = data["result"]
        # The pair key in the response may differ (e.g. "EURUSD" or "ZEURZUSD")
        pair_key = next(k for k in result if k != "last")
        trades = result[pair_key]
        since = int(result["last"])  # nanosecond cursor for next page

        if not trades:
            break

        for trade in trades:
            # Trade format: [price, volume, time, side, order_type, misc, trade_id]
            ts_sec = float(trade[2])
            ts_ms = int(ts_sec * 1000)

            if ts_ms > end_ms:
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
        else:
            # Inner loop didn't break, check if last trade is past end
            last_ts_ms = int(float(trades[-1][2]) * 1000)
            if last_ts_ms >= end_ms:
                break
            continue
        break  # Inner break triggered — we've passed end_ms

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
    cache_dir = Path(__file__).parent / 'cached_data' / exchange_name
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

    EXCHANGES = ["binance", "kraken", "dia"]

    # --- Sidebar config ---
    with st.sidebar:
        st.header("Settings")
        local_tz = st.text_input(
            "Timezone",
            value="America/Chicago",
            help="IANA timezone name, e.g. America/New_York, Europe/London"
        )
        n_buckets = st.slider(
            "Spread graph buckets",
            min_value=100, max_value=1000, value=400, step=50,
            help="The time range is divided into this many equal buckets. Each bucket "
                 "contributes at most one max-positive and one max-negative point, so "
                 "total rendered points ≤ 2 × buckets."
        )

    # --- Exchange + date range selectors ---
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        exchange1 = st.selectbox("Exchange A", EXCHANGES, index=0)
    with col2:
        exchange2 = st.selectbox("Exchange B", EXCHANGES, index=1)
    with col3:
        date_range = st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=1), date.today() - timedelta(days=1)),
            max_value=date.today() - timedelta(days=1),
        )

    # Validate date range
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_day, end_day = date_range
    else:
        st.info("Select a start and end date to begin.")
        st.stop()

    days_in_range = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]
    file1_label = exchange1.capitalize()
    file2_label = exchange2.capitalize()

    if st.button("Fetch Data", type="primary"):
        st.session_state.pop("df1", None)
        st.session_state.pop("df2", None)

        with st.spinner(f"Fetching {file1_label} data..."):
            try:
                df1 = pd.concat(
                    [get_prices_for_day(exchange1, d) for d in days_in_range],
                    ignore_index=True
                )
                st.session_state["df1"] = df1
            except Exception as e:
                st.error(f"Failed to fetch {file1_label} data: {e}")
                st.stop()

        with st.spinner(f"Fetching {file2_label} data..."):
            try:
                df2 = pd.concat(
                    [get_prices_for_day(exchange2, d) for d in days_in_range],
                    ignore_index=True
                )
                st.session_state["df2"] = df2
            except Exception as e:
                st.error(f"Failed to fetch {file2_label} data: {e}")
                st.stop()

    if "df1" not in st.session_state or "df2" not in st.session_state:
        st.info("Select exchanges and a date range, then click **Fetch Data**.")
        st.stop()

    # --- Validate timezone ---
    try:
        ZoneInfo(local_tz)
    except (ZoneInfoNotFoundError, KeyError):
        st.error(f"Unknown timezone: '{local_tz}'. Use an IANA name like 'America/Chicago'.")
        st.stop()

    # --- Load from session state ---
    @st.cache_data
    def prepare_df(df, tz):
        # Data is already in the correct format from get_prices_for_day,
        # but we still run it through detect_and_load for normalisation + tz handling
        return detect_and_load(df, tz)

    try:
        df1 = prepare_df(st.session_state["df1"], local_tz)
        df2 = prepare_df(st.session_state["df2"], local_tz)
    except Exception as e:
        st.error(f"Failed to prepare data: {e}")
        st.stop()

    # --- Everything below is unchanged from your original ---
    merged, overlap_start, overlap_end = build_merged(df1, df2)

    if merged.empty:
        st.warning("No overlapping time range found between the two sources.")
        st.stop()

    merged["spread_pct"] = ((merged["price1"] - merged["price2"]) / merged["price2"] * 100).round(4)

    hover_t, hover_v = decimate_spread(
        merged["time"].tolist(),
        merged["spread_pct"].tolist(),
        n_buckets,
    )

    segments = zero_crossing_segments(hover_t, hover_v)

    max_diff_row = merged.loc[merged["spread_pct"].abs().idxmax()]
    st.markdown(
        f"**{len(merged):,}** data points &nbsp;|&nbsp; "
        f"Max spread: **{max_diff_row['spread_pct']:+.4f}%** at `{max_diff_row['time'].strftime('%Y-%m-%d %H:%M:%S UTC')}`"
        f" &nbsp;|&nbsp; Mean: **{merged['spread_pct'].mean():+.4f}%**"
        f" &nbsp;|&nbsp; Std: **{merged['spread_pct'].std():.4f}%**"
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Price", "Spread (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=df1["time"], y=df1["price"],
            name=file1_label,
            line=dict(color="#4C9BE8", width=1.2, shape="hv"),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>" + file1_label + "</extra>",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df2["time"], y=df2["price"],
            name=file2_label,
            line=dict(color="#F4A83A", width=1.2, dash="dot", shape="hv"),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>" + file2_label + "</extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hover_t, y=hover_v,
            name="Spread",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Spread: %{y:+.4f}%<extra></extra>",
        ),
        row=2, col=1
    )

    shown_pos = shown_neg = False
    for seg_t, seg_v, is_pos in segments:
        color = "#2ECC71" if is_pos else "#E74C3C"
        name  = "Spread (A > B)" if is_pos else "Spread (A < B)"
        show  = (is_pos and not shown_pos) or (not is_pos and not shown_neg)
        fig.add_trace(
            go.Scatter(
                x=seg_t, y=seg_v,
                name=name,
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=show,
                legendgroup=name,
                hoverinfo="skip",
            ),
            row=2, col=1
        )
        if is_pos: shown_pos = True
        else:      shown_neg = True

    max_pos_idx = merged["spread_pct"].idxmax()
    max_neg_idx = merged["spread_pct"].idxmin()
    for idx, label_color in [(max_pos_idx, "#2ECC71"), (max_neg_idx, "#E74C3C")]:
        row = merged.loc[idx]
        fig.add_trace(
            go.Scatter(
                x=[row["time"]], y=[row["spread_pct"]],
                mode="markers+text",
                marker=dict(color=label_color, size=8, symbol="circle"),
                text=[f"{row['spread_pct']:+.4f}%"],
                textposition="top center" if row["spread_pct"] >= 0 else "bottom center",
                textfont=dict(color=label_color, size=11),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.update_yaxes(title_text="EUR/USD", row=1, col=1, tickformat=".5f")
    fig.update_yaxes(title_text="Spread %", row=2, col=1, tickformat="+.4f%")
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View merged data table"):
        display = merged.copy()
        display["time"]       = display["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        display["price1"]     = display["price1"].map("{:.6f}".format)
        display["price2"]     = display["price2"].map("{:.6f}".format)
        display["spread_pct"] = display["spread_pct"].map("{:+.4f}%".format)
        display.columns = ["Time (UTC)", file1_label, file2_label, "Spread %"]
        st.dataframe(display, use_container_width=True, hide_index=True)
