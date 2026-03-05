import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dateutil
from datetime import date, datetime, timedelta, timezone, time
from pathlib import Path
from io import StringIO

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.fixed_price import FixedPriceExchange
from hydradx.model.amm.agents import Agent


# =============================================================================
# CLOUD STORAGE (S3 / S3-compatible)
# =============================================================================
# Configure via Streamlit secrets (secrets.toml) or environment variables:
#
#   [s3]
#   bucket      = "my-arb-sim-cache"
#   prefix      = "price-cache/"          # optional, default ""
#   aws_access_key_id     = "..."
#   aws_secret_access_key = "..."
#   region_name           = "us-east-1"   # optional
#   endpoint_url          = "..."         # optional — for R2, MinIO, etc.
#
# If none of the above are set, cloud storage is silently disabled and the
# app behaves exactly as before (local disk cache only).
# =============================================================================

def _s3_config() -> dict | None:
    """
    Return S3 config dict from st.secrets or environment variables,
    or None if cloud storage is not configured.
    """
    # Try Streamlit secrets first
    try:
        cfg = st.secrets.get("s3", {})
        bucket = cfg.get("bucket") or os.environ.get("S3_BUCKET")
    except Exception:
        bucket = os.environ.get("S3_BUCKET")
        cfg = {}

    if not bucket:
        return None  # cloud storage not configured — that's fine

    def _get(key: str, env_key: str | None = None) -> str | None:
        val = cfg.get(key)
        if val:
            return val
        return os.environ.get(env_key or key.upper())

    return {
        "bucket": bucket,
        "prefix": _get("prefix", "S3_PREFIX") or "",
        "aws_access_key_id": _get("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": _get("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
        "region_name": _get("region_name", "AWS_DEFAULT_REGION"),
        "endpoint_url": _get("endpoint_url", "S3_ENDPOINT_URL"),  # for R2 / MinIO
    }


@st.cache_resource(show_spinner=False)
def _get_s3_client():
    """Return a boto3 S3 client (cached for the session), or None."""
    cfg = _s3_config()
    if cfg is None:
        return None, None

    try:
        import boto3

        client_kwargs = {}
        for key in ("aws_access_key_id", "aws_secret_access_key", "region_name", "endpoint_url"):
            if cfg.get(key):
                client_kwargs[key] = cfg[key]

        client = boto3.client("s3", **client_kwargs)
        return client, cfg
    except ImportError:
        st.warning("boto3 is not installed — cloud cache disabled. Run `pip install boto3`.")
        return None, None
    except Exception as exc:
        st.warning(f"Could not initialise S3 client — cloud cache disabled: {exc}")
        return None, None


def _s3_key(exchange: str, day: date, cfg: dict) -> str:
    prefix = cfg["prefix"].rstrip("/")
    key = f"{exchange}/{day.isoformat()}.csv"
    return f"{prefix}/{key}" if prefix else key


def _download_from_s3(exchange: str, day: date) -> pd.DataFrame | None:
    """
    Try to download a cached CSV from S3. Returns a DataFrame on success,
    None if the object doesn't exist or cloud storage is unconfigured.
    """
    client, cfg = _get_s3_client()
    if client is None:
        return None

    key = _s3_key(exchange, day, cfg)
    try:
        response = client.get_object(Bucket=cfg["bucket"], Key=key)
        body = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(body))
        print(f"[cloud] Downloaded {exchange}/{day} from s3://{cfg['bucket']}/{key}")
        return df
    except client.exceptions.NoSuchKey:
        return None  # not cached yet — fetch from exchange API
    except Exception as exc:
        print(f"[cloud] S3 download failed for {key}: {exc}")
        return None


def _upload_to_s3(df: pd.DataFrame, exchange: str, day: date) -> None:
    """Upload a price DataFrame as CSV to S3. Silently skips on any error."""
    client, cfg = _get_s3_client()
    if client is None:
        return

    key = _s3_key(exchange, day, cfg)
    try:
        body = df.to_csv(index=False).encode("utf-8")
        client.put_object(Bucket=cfg["bucket"], Key=key, Body=body, ContentType="text/csv")
        print(f"[cloud] Uploaded {exchange}/{day} to s3://{cfg['bucket']}/{key}")
    except Exception as exc:
        print(f"[cloud] S3 upload failed for {key}: {exc}")


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

        if ts >= start_ms:
            if interval_ms:
                if ts < current_interval:
                    continue
                current_interval += interval_ms
            binance_rows.append({
                "timestamp_ms": ts,
                "readable_time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[
                                 :-3],
                "pair": "EURUSD",
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
    """
    Load price data for one exchange/day using a three-tier cache:
      1. Local disk  — instant, survives app restarts on the same machine
      2. S3 / R2     — fast, shared across all deployed instances
      3. Exchange API — source of truth, result is written back to both caches
    """
    cache_dir = Path(__file__).parent / 'cached data' / exchange_name
    cache_file = cache_dir / f'{day.strftime("%Y-%m-%d")}.csv'

    # ── Tier 1: local disk ───────────────────────────────────────────────────
    if cache_file.exists():
        print(f"[local] Loading cached {exchange_name} data for {day}")
        return pd.read_csv(cache_file)

    # ── Tier 2: cloud (S3 / R2 / MinIO) ─────────────────────────────────────
    cloud_df = _download_from_s3(exchange_name, day)
    if cloud_df is not None:
        # Populate local disk so the next hit is even faster
        cache_dir.mkdir(parents=True, exist_ok=True)
        cloud_df.to_csv(cache_file, index=False)
        return cloud_df

    # ── Tier 3: exchange API ─────────────────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.combine(day, time.min, tzinfo=timezone.utc)

    fetchers = {
        'binance': get_binance_prices,
        'kraken': get_kraken_prices,
    }
    if exchange_name not in fetchers:
        raise ValueError(f"Unknown exchange '{exchange_name}'. Expected one of: {list(fetchers)}")

    df = fetchers[exchange_name](start_date=start_dt, days=1, save_path=cache_file)

    # Write to cloud so other instances (or future deployments) skip the API call
    _upload_to_s3(df, exchange_name, day)

    return df


def _load_prices(exchange: str, day_keys: tuple[str, ...]) -> pd.DataFrame:
    days = [date.fromisoformat(k) for k in day_keys]
    frames = [get_prices_for_day(exchange, d) for d in days]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_prices_cached(exchange: str, day_keys: tuple[str, ...]) -> pd.DataFrame:
    return _load_prices(exchange, day_keys)


@st.cache_data(show_spinner=False)
def load_dia_cached() -> pd.DataFrame:
    dia_path = Path(__file__).parent / "DIA" / "DIA_data.csv"
    if not dia_path.exists():
        return pd.DataFrame(columns=["timestamp_ms", "time", "price"])

    df = pd.read_csv(dia_path, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    if "timestamp" not in df.columns or "EUR/USD" not in df.columns:
        return pd.DataFrame(columns=["timestamp_ms", "time", "price"])

    times = pd.to_datetime(df["timestamp"], errors="coerce")
    if times.dt.tz is None:
        times = times.dt.tz_localize("US/Central", ambiguous="NaT", nonexistent="NaT")
    else:
        times = times.dt.tz_convert("US/Central")
    times = times.dt.tz_convert("UTC")

    prices = pd.to_numeric(df["EUR/USD"].astype(str).str.replace("$", "", regex=False), errors="coerce")
    out = pd.DataFrame({"time": times, "price": prices}).dropna().sort_values("time")
    out["timestamp_ms"] = (out["time"].astype("int64") // 1_000_000).astype("int64")
    return out[["timestamp_ms", "time", "price"]]


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


def build_simulation_points(
    combined_df: pd.DataFrame,
    dia_df: pd.DataFrame,
    step_seconds: int = 6,
) -> list[dict]:
    if combined_df.empty:
        return []

    combined = combined_df[["time", "external_price"]].dropna().copy()
    combined["timestamp_ms"] = (combined["time"].astype("int64") // 1_000_000).astype("int64")

    if dia_df.empty:
        start_dt = combined["time"].min()
        end_dt = combined["time"].max()
    else:
        start_dt = max(combined["time"].min(), dia_df["time"].min())
        end_dt = min(combined["time"].max(), dia_df["time"].max())

    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt >= end_dt:
        return []

    grid = pd.date_range(start=start_dt, end=end_dt, freq=f"{step_seconds}S", tz="UTC")
    grid_ms = (grid.astype("int64") // 1_000_000).astype("int64")

    combined_interp = _interp_prices(
        grid_ms,
        combined["timestamp_ms"].to_numpy(dtype=np.int64),
        combined["external_price"].to_numpy(dtype=float),
    )

    if dia_df.empty:
        dia_interp = np.full_like(combined_interp, np.nan, dtype=float)
    else:
        dia_interp = _interp_prices(
            grid_ms,
            dia_df["timestamp_ms"].to_numpy(dtype=np.int64),
            dia_df["price"].to_numpy(dtype=float),
        )

    return [
        {
            "time": t,
            "timestamp_ms": int(ms),
            "external_price": float(sp),
            "dia_price": float(dp) if not np.isnan(dp) else np.nan,
        }
        for t, ms, sp, dp in zip(grid, grid_ms, combined_interp, dia_interp)
    ]


def run_sim(
        steps: list[dict[str, float]],
        trade_fee: float=0.0,
        amplification: float=100.0
) -> dict[datetime, float]:
    dia_start_peg = steps[0]["dia_price"]
    eur_usd_stablepool = StableSwapPoolState(
        tokens={"USD": 1_000_000 * dia_start_peg, "EUR": 1_000_000},
        amplification=amplification,
        trade_fee=trade_fee,
        peg=dia_start_peg,
        spot_price_precision=0.00000000001,
        precision=1e-12
    )
    binance = FixedPriceExchange(
        tokens={"EUR": steps[0]["external_price"], "USD": 1.0},
    )
    profit_over_time = {}
    arbitrageur = Agent()
    for i, step in enumerate(steps):
        next_peg = step["dia_price"]
        eur_usd_stablepool.set_peg_target(next_peg)
        eur_usd_stablepool.update()
        binance.prices = {"EUR": step["external_price"], "USD": 1.0}
        max_arb_size = 2 ** 10
        arb_size = max_arb_size / 2
        max_iterations = 10
        spread = abs(1 - step['external_price'] / step['dia_price'])
        if spread < eur_usd_stablepool.trade_fee:
            continue
        buy_or_sell = "buy" if step["external_price"] > eur_usd_stablepool.price("EUR", "USD") else "sell"
        best = None
        for j in range(2, max_iterations + 2):
            delta = max_arb_size / (2 ** j)
            try_arbs = [arb_size - delta, arb_size + delta]
            trade_opts = [
                {
                    "usd": -try_arb if buy_or_sell == "buy" else try_arb,
                    "eur": eur_usd_stablepool.calculate_buy_from_sell(
                            tkn_buy="EUR", tkn_sell="USD", sell_quantity=try_arb
                        ) if buy_or_sell == "buy" else -eur_usd_stablepool.calculate_sell_from_buy(
                            tkn_sell="EUR", tkn_buy="USD", buy_quantity=try_arb
                    )} for try_arb in try_arbs
            ]
            if best:
                trade_opts.append(best)
            results = [
                {
                    "usd": trade["usd"],
                    "eur": trade["eur"],
                    "profit": step["external_price"] * trade["eur"] + trade["usd"]
                } for trade in trade_opts
            ]
            best = max(results, key=lambda item: item["profit"])
            arb_size = abs(best['usd'])
        if best["profit"] > 0.1:
            start_usd = arbitrageur.get_holdings('USD')
            if buy_or_sell == "buy":
                eur_usd_stablepool.swap(
                    agent=arbitrageur,
                    tkn_sell="USD",
                    tkn_buy="EUR",
                    sell_quantity=arb_size
                )
                if abs(arbitrageur.get_holdings('EUR') - best['eur']) > 0.00001:
                    print("arbitrage calculation is off")
                binance.swap(
                    arbitrageur, tkn_sell='EUR', tkn_buy='USD', sell_quantity=arbitrageur.holdings['EUR']
                )
            else:
                eur_usd_stablepool.swap(
                    agent=arbitrageur,
                    tkn_sell="EUR",
                    tkn_buy="USD",
                    buy_quantity=arb_size
                )
                if (abs(arbitrageur.get_holdings('EUR') - best['eur']) > 0.00001):
                    print("arbitrage calculation is off")
                binance.swap(
                    arbitrageur, tkn_buy='EUR', tkn_sell='USD', buy_quantity=-arbitrageur.holdings['EUR']
                )
            profit = arbitrageur.get_holdings('USD') - start_usd
            profit_over_time[datetime.fromtimestamp(step["timestamp_ms"] / 1000)] = profit
            if profit < 0:
                raise ValueError("arbitrage messed up")
        else:
            continue

    profit = arbitrageur.get_holdings('USD')
    return profit_over_time


def smooth_binance_with_kraken(
    binance_df: pd.DataFrame,
    kraken_df: pd.DataFrame,
    binance_bias_factor: float = 3.0,
) -> pd.DataFrame:
    binance = _to_price_df(binance_df)
    kraken = _to_price_df(kraken_df)

    if binance.empty:
        return pd.DataFrame(columns=[
            "timestamp_ms", "time", "binance_price", "kraken_price", "external_price"
        ])

    target_ms = binance["timestamp_ms"].to_numpy(dtype=np.int64)
    b_prices = binance["price"].to_numpy(dtype=float)

    k_times = kraken["timestamp_ms"].to_numpy(dtype=np.int64)
    k_prices = kraken["price"].to_numpy(dtype=float)

    k_interp = _interp_prices(target_ms, k_times, k_prices)

    combined = []
    last_value = float(b_prices[0])
    denom = max(abs(binance_bias_factor), 1e-6)
    for b_price, k_price in zip(b_prices, k_interp):
        if np.isnan(k_price):
            chosen = b_price
        else:
            b_diff = abs(b_price - last_value) / denom
            k_diff = abs(k_price - last_value)
            if b_diff <= k_diff:
                chosen = b_price
            else:
                chosen = k_price
        combined.append(chosen)
        last_value = chosen

    out = pd.DataFrame({
        "timestamp_ms": binance["timestamp_ms"],
        "time": binance["time"],
        "binance_price": b_prices,
        "kraken_price": k_interp,
        "external_price": np.array(combined, dtype=float),
    })
    return out


# =============================================================================
# STREAMLIT UI
# =============================================================================
if st.runtime.exists():
    st.set_page_config(page_title="EUR/USD Arbitrage Simulation", layout="wide")
    st.title("EUR/USD Arbitrage Simulation")

    # Show cloud storage status unobtrusively in the sidebar
    _client, _cfg = _get_s3_client()
    sidebar = st.sidebar.container()
    if _cfg:
        sidebar.success(f"☁️ Cloud cache: `{_cfg['bucket']}/{_cfg['prefix']}`", icon=None)
    else:
        sidebar.info("☁️ Cloud cache: not configured (local only)", icon=None)

    sidebar.header("Settings")

    date_range = sidebar.date_input(
        "Date range",
        value=(date.today() - timedelta(days=1), date.today() - timedelta(days=1)),
        max_value=date.today() - timedelta(days=1),
        key="date_range",
    )

    if not (isinstance(date_range, (list, tuple)) and len(date_range) == 2):
        st.info("Select a start and end date to begin.")
        st.stop()

    start_day, end_day = date_range
    range_key = (start_day.isoformat(), end_day.isoformat())

    if st.session_state.get("selected_range_key") != range_key:
        st.session_state["selected_range_key"] = range_key
        st.session_state["data_ready"] = False
        st.session_state["simulation_ran"] = False
        st.session_state["simulation_results"] = None
        st.session_state["binance_df"] = pd.DataFrame()
        st.session_state["kraken_df"] = pd.DataFrame()
        st.session_state["dia_df"] = pd.DataFrame()

    if not st.session_state.get("data_ready", False):
        days_in_range = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]
        day_keys = tuple(d.isoformat() for d in days_in_range)

        with st.spinner("Loading Binance data..."):
            try:
                binance_df = load_prices_cached("binance", day_keys)
            except Exception as exc:
                st.error(f"Failed to fetch Binance data: {exc}")
                st.stop()

        with st.spinner("Loading Kraken data..."):
            try:
                kraken_df = load_prices_cached("kraken", day_keys)
            except Exception as exc:
                st.error(f"Failed to fetch Kraken data: {exc}")
                st.stop()

        dia_df = load_dia_cached()
        dia_range = None
        if not dia_df.empty:
            dia_range = (dia_df["time"].min(), dia_df["time"].max())
            start_dt = pd.Timestamp(start_day, tz="UTC")
            end_dt = pd.Timestamp(end_day + timedelta(days=1), tz="UTC")
            dia_df = dia_df[(dia_df["time"] >= start_dt) & (dia_df["time"] < end_dt)]

        st.session_state["dia_range"] = dia_range
        st.session_state["day_keys"] = day_keys
        st.session_state["binance_df"] = binance_df
        st.session_state["kraken_df"] = kraken_df
        st.session_state["dia_df"] = dia_df
        st.session_state["data_ready"] = True

    binance_df = st.session_state["binance_df"]
    kraken_df = st.session_state["kraken_df"]
    dia_df = st.session_state["dia_df"]
    dia_range = st.session_state.get("dia_range")

    if binance_df.empty or kraken_df.empty:
        st.warning("No price data available for the selected range.")
        st.stop()

    if dia_df.empty:
        if dia_range is not None:
            st.warning(
                "No DIA data in the selected range. Available DIA range: "
                f"{dia_range[0].strftime('%Y-%m-%d %H:%M:%S UTC')} to "
                f"{dia_range[1].strftime('%Y-%m-%d %H:%M:%S UTC')}."
            )
        else:
            st.warning("No DIA oracle data available. Simulation requires DIA data.")
        st.stop()

    binance_bias_factor = sidebar.slider(
        "Binance bias factor",
        min_value=1.0, max_value=10.0,
        value=st.session_state.get("binance_bias_factor", 3.0),
        step=0.1,
        help="How closely the simulation price tracks Binance as opposed to Kraken.",
        key="binance_bias_factor",
    )
    pool_amplification = sidebar.slider(
        "StableSwap amplification factor",
        min_value=10, max_value=500,
        value=st.session_state.get("pool_amplification", 100),
        step=10, key="pool_amplification",
    )
    trade_fee = sidebar.slider(
        "StableSwap trade fee",
        min_value=0.0, max_value=0.001,
        value=st.session_state.get("trade_fee", 0.0005),
        step=0.0001, key="trade_fee", format="%.4f",
    )

    run_sim_clicked = sidebar.button("Run simulation", type="primary")

    combined_df = smooth_binance_with_kraken(binance_df, kraken_df, binance_bias_factor)

    if not combined_df.empty:
        dia_norm = _to_price_df(dia_df) if not dia_df.empty else pd.DataFrame(columns=["timestamp_ms", "time", "price"])

        toggle_cols = st.columns(4)
        show_binance  = toggle_cols[0].checkbox("Binance",  value=True, key="show_binance")
        show_kraken   = toggle_cols[1].checkbox("Kraken",   value=True, key="show_kraken")
        show_combined = toggle_cols[2].checkbox("Combined", value=True, key="show_combined")
        show_dia      = toggle_cols[3].checkbox("DIA",      value=True, key="show_dia")

        fig = go.Figure()
        if show_binance:
            fig.add_trace(go.Scatter(
                x=combined_df["time"], y=combined_df["binance_price"],
                name="Binance", line=dict(color="#58D68D", width=1.2),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Binance</extra>",
            ))
        if show_kraken:
            fig.add_trace(go.Scatter(
                x=combined_df["time"], y=combined_df["kraken_price"],
                name="Kraken (interp)", line=dict(color="#F4D03F", width=1.2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Kraken</extra>",
            ))
        if show_combined:
            fig.add_trace(go.Scatter(
                x=combined_df["time"], y=combined_df["external_price"],
                name="combined", line=dict(color="#E74C3C", width=1.6),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>combined</extra>",
            ))
        if show_dia and not dia_norm.empty:
            fig.add_trace(go.Scatter(
                x=dia_norm["time"], y=dia_norm["price"],
                name="DIA", line=dict(color="#B86BFF", width=1.4, dash="dot", shape="hv"),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>DIA</extra>",
            ))
        fig.update_layout(
            height=500, template="plotly_dark", hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0),
            margin=dict(t=60, b=40, l=60, r=20),
        )
        fig.update_yaxes(title_text="EUR/USD", tickformat=".5f")
        fig.update_xaxes(title_text="Time (UTC)")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    sim_param_key = (binance_bias_factor, pool_amplification, trade_fee)
    if st.session_state.get("sim_param_key") != sim_param_key:
        st.session_state["sim_param_key"] = sim_param_key
        st.session_state["simulation_ran"] = False
        st.session_state["simulation_results"] = None

    if run_sim_clicked:
        simulation_points = build_simulation_points(combined_df, dia_df, step_seconds=6)

        if not simulation_points:
            st.warning("Could not build simulation points — check that price and DIA data overlap in time.")
        else:
            with st.spinner("Running simulation..."):
                sim_results = run_sim(
                    steps=simulation_points,
                    trade_fee=trade_fee,
                    amplification=pool_amplification,
                )
            st.session_state["simulation_results"] = sim_results
            st.session_state["simulation_ran"] = True

    if not st.session_state.get("simulation_ran"):
        st.info("Configure the parameters in the sidebar, then click **Run simulation**.")
    else:
        sim_results = st.session_state.get("simulation_results") or {}
        if not sim_results:
            st.info("Simulation completed, but no profitable arbitrage events were found.")
        else:
            profit_df = (
                pd.DataFrame(
                    {"time": list(sim_results.keys()), "profit": list(sim_results.values())}
                )
                .sort_values("time")
            )
            sim_end = st.session_state.get("simulation_end_time")
            if sim_end is not None and (profit_df.empty or sim_end > profit_df["time"].iloc[-1]):
                sentinel = pd.DataFrame({"time": [sim_end], "profit": [0.0]})
                profit_df = pd.concat([profit_df, sentinel], ignore_index=True)
            profit_df["cumulative_profit"] = profit_df["profit"].cumsum()

            total = profit_df["cumulative_profit"].iloc[-1]
            n_events = len(profit_df)
            col1, col2 = st.columns(2)
            col1.metric("Total arbitrage profit (USD)", f"${total:,.4f}")
            col2.metric("Arbitrage events", f"{n_events:,}")

            profit_fig = go.Figure()
            profit_fig.add_trace(
                go.Scatter(
                    x=profit_df["time"],
                    y=profit_df["cumulative_profit"],
                    name="Arbitrageur profit (cumulative)",
                    line=dict(color="#00B3FF", width=2.0),
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Total Profit: %{y:.4f}<extra></extra>",
                )
            )
            profit_fig.update_layout(
                height=500, template="plotly_dark", hovermode="x unified",
                margin=dict(t=40, b=40, l=60, r=20),
            )
            profit_fig.update_yaxes(title_text="USD Profit (Cumulative)", tickformat=".4f")
            profit_fig.update_xaxes(title_text="Time (UTC)")
            st.plotly_chart(profit_fig, use_container_width=True)


# =============================================================================
# CLI DEMO
# =============================================================================
if __name__ == "__main__" and not st.runtime.exists():
    demo_day = date.today() - timedelta(days=1)
    binance_demo = get_prices_for_day("binance", demo_day)
    kraken_demo = get_prices_for_day("kraken", demo_day)
    demo = smooth_binance_with_kraken(binance_demo, kraken_demo, binance_bias_factor=3.0)
    print(demo.head(5).to_string(index=False))