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

from sympy import false, true

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


def _step_prices(target_ms: np.ndarray, source_ms: np.ndarray, source_prices: np.ndarray) -> np.ndarray:
    if len(source_ms) == 0:
        return np.full_like(target_ms, np.nan, dtype=float)
    if len(source_ms) == 1:
        return np.full_like(target_ms, float(source_prices[0]), dtype=float)

    order = np.argsort(source_ms)
    src_ms = source_ms[order]
    src_prices = source_prices[order]

    idx = np.searchsorted(src_ms, target_ms, side="right") - 1
    out = np.full_like(target_ms, np.nan, dtype=float)
    valid = (idx >= 0) & (idx < len(src_prices))
    out[valid] = src_prices[idx[valid]]
    return out


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
        dia_interpolated = false
        if dia_interpolated:
            dia_interp = _interp_prices(
                grid_ms,
                dia_df["timestamp_ms"].to_numpy(dtype=np.int64),
                dia_df["price"].to_numpy(dtype=float),
            )
        else:
            dia_interp = _step_prices(
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
        amplification: float=100.0,
        return_series: bool=False,
) -> dict[datetime, float] | tuple[dict[datetime, float], pd.DataFrame]:
    dia_start_peg = steps[0]["dia_price"]
    eur_usd_stablepool = StableSwapPoolState(
        tokens={"USD": 1_000_000 * dia_start_peg, "EUR": 1_000_000},
        amplification=amplification,
        trade_fee=trade_fee,
        peg=dia_start_peg,
        spot_price_precision=0.00000000001,
        precision=1e-12,
        max_peg_update=0.0001
    )
    binance = FixedPriceExchange(
        tokens={"EUR": steps[0]["external_price"], "USD": 1.0},
    )
    profit_over_time = {}
    series_times: list[datetime] = []
    series_external: list[float] = []
    series_dia: list[float] = []
    series_stableswap: list[float] = []
    series_peg: list[float] = []
    series_peg_target: list[float] = []
    last_hour_logged: datetime | None = None
    arbitrageur = Agent()
    trader = Agent()
    for i, step in enumerate(steps):
        next_peg = step["dia_price"]
        eur_usd_stablepool.set_peg_target(next_peg)
        eur_usd_stablepool.update()
        binance.prices = {"EUR": step["external_price"], "USD": 1.0}

        # do one random $1 trade to update the price and trigger peg adjustments if needed
        buy_or_sell = "buy" if step["external_price"] > step["dia_price"] else "sell"
        eur_usd_stablepool.swap(
            agent=trader,
            tkn_sell="USD" if buy_or_sell == "buy" else "EUR",
            tkn_buy="EUR" if buy_or_sell == "buy" else "USD",
            sell_quantity=0.00001
        )

        stableswap_price_before = float(eur_usd_stablepool.price("EUR", "USD"))
        hour_key = step["time"].replace(minute=0, second=0, microsecond=0)
        if last_hour_logged is None or hour_key > last_hour_logged:
            last_hour_logged = hour_key
            print(
                f"[hourly] {hour_key.isoformat()} "
                f"ext={step['external_price']:.6f} "
                f"dia={step['dia_price']:.6f} "
                f"stableswap={stableswap_price_before:.6f}"
            )

        max_arb_size = 2 ** 10
        arb_size = max_arb_size / 2
        max_iterations = 10
        spread = abs(1 - step['external_price'] / step['dia_price'])
        if spread >= eur_usd_stablepool.trade_fee:
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
                profit_over_time[datetime.fromtimestamp(step["timestamp_ms"] / 1000, tz=timezone.utc)] = profit
                stableswap_price_after = float(eur_usd_stablepool.price("EUR", "USD"))
                print(
                    f"[arb] {step['time'].isoformat()} "
                    f"ext={step['external_price']:.6f} "
                    f"dia={step['dia_price']:.6f} "
                    f"stableswap_before={stableswap_price_before:.6f} "
                    f"arb_usd={arb_size:.6f} "
                    f"stableswap_after={stableswap_price_after:.6f}"
                )
                if profit < 0:
                    raise ValueError("arbitrage messed up")

        series_times.append(step["time"])
        series_external.append(float(step["external_price"]))
        series_dia.append(float(step["dia_price"]))
        series_stableswap.append(float(eur_usd_stablepool.price("EUR", "USD")))
        series_peg.append(float(eur_usd_stablepool.peg[1]))
        series_peg_target.append(float(eur_usd_stablepool.peg_target[1]))

    if return_series:
        series_df = pd.DataFrame({
            "time": series_times,
            "external_price": series_external,
            "dia_price": series_dia,
            "stableswap_price": series_stableswap,
            "stableswap_peg": series_peg,
            "stableswap_peg_target": series_peg_target,
        })
        return profit_over_time, series_df
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


def _downsample_for_plot(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def _pad_series_to_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    series = df.sort_values("time").copy()
    series["time"] = pd.to_datetime(series["time"], utc=True)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    in_range = series[(series["time"] >= start) & (series["time"] <= end)].copy()
    before_start = series[series["time"] < start].tail(1)
    after_end = series[series["time"] > end].head(1)

    pieces = []
    if not before_start.empty:
        start_row = before_start.copy()
        start_row["time"] = start
        start_row["timestamp_ms"] = int(start.value // 1_000_000)
        pieces.append(start_row)

    if not in_range.empty:
        pieces.append(in_range)

    if not after_end.empty:
        end_row = after_end.copy()
        end_row["time"] = end
        end_row["timestamp_ms"] = int(end.value // 1_000_000)
        pieces.append(end_row)

    if not pieces:
        return in_range

    window = pd.concat(pieces, ignore_index=True)
    return window.drop_duplicates(subset=["time"]).sort_values("time")


# =============================================================================
# STREAMLIT FRAGMENT
# =============================================================================
@st.fragment
def simulation_section() -> None:
    combined_df = st.session_state.get("combined_df", pd.DataFrame())
    dia_df = st.session_state.get("dia_df", pd.DataFrame())

    if combined_df.empty or dia_df.empty:
        st.info("Configure the parameters in the sidebar, then click **Run simulation**.")
        return

    # Sim controls live in the main body, not the sidebar
    col_a, col_b = st.columns(2)
    pool_amplification = col_a.slider(
        "StableSwap amplification factor",
        min_value=10, max_value=500,
        value=st.session_state.get("pool_amplification", 100),
        step=10, key="pool_amplification",
    )
    trade_fee = col_b.slider(
        "StableSwap trade fee",
        min_value=0.0, max_value=0.001,
        value=st.session_state.get("trade_fee", 0.0005),
        step=0.0001, key="trade_fee", format="%0.4f"
    )

    # Small preview panel that refreshes only when StableSwap settings change.
    preview_box = col_b.container()
    preview_row = preview_box.columns([2, 1, 2, 2])
    preview_row[0].markdown("Price of a")
    usd_trade_size_raw = preview_row[1].text_input(
        "USD trade size",
        value=st.session_state.get("preview_usd_trade_size", "10000"),
        key="preview_usd_trade_size",
        label_visibility="collapsed",
    )
    preview_row[2].markdown("USD trade:")
    preview_value_slot = preview_row[3].empty()
    try:
        usd_trade_size = float(usd_trade_size_raw)
    except (TypeError, ValueError):
        usd_trade_size = None

    preview_key = (
        pool_amplification,
        trade_fee,
        st.session_state.get("pool_depth", 2_000_000),
        usd_trade_size,
    )
    preview_data_key = st.session_state.get("day_keys")
    preview_needs_update = (
        st.session_state.get("preview_key") != preview_key
        or st.session_state.get("preview_data_key") != preview_data_key
        or st.session_state.get("preview_result") is None
    )

    if preview_needs_update and usd_trade_size is not None and not combined_df.empty and not dia_df.empty:
        with preview_box:
            with st.spinner("Updating preview..."):
                combined_small = combined_df[["time", "external_price"]].dropna().sort_values("time")
                dia_small = dia_df[["time", "price"]].dropna().sort_values("time")

                preview_start = max(combined_small["time"].min(), dia_small["time"].min())
                dia_overlap = dia_small[dia_small["time"] >= preview_start]
                if dia_overlap.empty:
                    preview_payload = None
                else:
                    dia_price = float(dia_overlap["price"].iloc[0])
                    eur_usd_stableswap = StableSwapPoolState(
                        tokens={"USD": 1_000_000 * dia_price, "EUR": 1_000_000},
                        amplification=pool_amplification,
                        trade_fee=trade_fee,
                        peg=dia_price,
                        spot_price_precision=0.00000000001,
                        precision=1e-12,
                        max_peg_update=0.0001,
                    )
                    pool_after = eur_usd_stableswap.copy()
                    pool_after.swap(
                        agent=Agent(),
                        tkn_buy="EUR",
                        tkn_sell="USD",
                        sell_quantity=usd_trade_size,
                    )
                    eur_received = (
                        eur_usd_stableswap.liquidity["EUR"]
                        - pool_after.liquidity["EUR"]
                    )
                    trade_value = eur_received * dia_price
                    cost = usd_trade_size - trade_value
                    preview_payload = {
                        "cost": float(cost),
                        "dia_price": dia_price,
                        "trade_size": usd_trade_size,
                    }

                st.session_state["preview_key"] = preview_key
                st.session_state["preview_data_key"] = preview_data_key
                st.session_state["preview_result"] = preview_payload

    preview_payload = st.session_state.get("preview_result") if usd_trade_size is not None else None
    if preview_payload is None:
        preview_value_slot.markdown("—")
    else:
        preview_value_slot.markdown(f"${preview_payload['cost']:,.6f} ({(preview_payload['cost'] / preview_payload['trade_size']):,.4f}%)")

    pool_depth = col_a.slider(
        "StableSwap total liquidity (USD)",
        min_value=100_000, max_value=10_000_000,
        value=st.session_state.get("pool_depth", 2_000_000),
        step=100_000, key="pool_depth"
    )
    run_sim_clicked = st.button("Run simulation", type="primary")

    # Invalidate results when sim params change
    sim_param_key = (pool_amplification, trade_fee)
    if st.session_state.get("sim_param_key") != sim_param_key:
        st.session_state["sim_param_key"] = sim_param_key
        st.session_state["simulation_ran"] = False
        st.session_state["simulation_results"] = None

    if run_sim_clicked:
        simulation_points = build_simulation_points(combined_df, dia_df, step_seconds=6)
        if not simulation_points:
            st.warning("Could not build simulation points — check that price and DIA data overlap in time.")
        else:
            st.session_state["simulation_start_time"] = simulation_points[0]["time"]
            st.session_state["simulation_end_time"] = simulation_points[-1]["time"]
            with st.spinner("Running simulation..."):
                sim_results, sim_series = run_sim(
                    steps=simulation_points,
                    trade_fee=trade_fee,
                    amplification=pool_amplification,
                    return_series=True,
                )
            st.session_state["simulation_results"] = sim_results
            st.session_state["simulation_series"] = sim_series
            st.session_state["simulation_ran"] = True

    if not st.session_state.get("simulation_ran"):
        st.info("Configure the parameters in the sidebar, then click **Run simulation**.")
    else:
        sim_results = st.session_state.get("simulation_results") or {}
        sim_start = st.session_state.get("simulation_start_time")
        sim_end = st.session_state.get("simulation_end_time")
        if not sim_results:
            if sim_start is None or sim_end is None:
                st.info("Simulation completed, but no profitable arbitrage events were found.")
                return
            profit_df = pd.DataFrame({"time": [sim_start, sim_end], "profit": [0.0, 0.0]})
        else:
            profit_df = (
                pd.DataFrame(
                    {"time": list(sim_results.keys()), "profit": list(sim_results.values())}
                )
                .sort_values("time")
            )
            if sim_start is not None and (profit_df.empty or sim_start < profit_df["time"].iloc[0]):
                start_sentinel = pd.DataFrame({"time": [sim_start], "profit": [0.0]})
                profit_df = pd.concat([start_sentinel, profit_df], ignore_index=True)
            if sim_end is not None and (profit_df.empty or sim_end > profit_df["time"].iloc[-1]):
                end_sentinel = pd.DataFrame({"time": [sim_end], "profit": [0.0]})
                profit_df = pd.concat([profit_df, end_sentinel], ignore_index=True)

        profit_df["cumulative_profit"] = profit_df["profit"].cumsum()

        total = profit_df["cumulative_profit"].iloc[-1]
        n_events = len(sim_results)
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
        profit_fig.update_yaxes(title_text="USD Profit (Cumulative)", tickformat=".2f")
        profit_fig.update_xaxes(title_text="Time (UTC)")
        st.plotly_chart(profit_fig, use_container_width=True)

        series_df = st.session_state.get("simulation_series")
        if isinstance(series_df, pd.DataFrame) and not series_df.empty:
            st.subheader("Spread between price sources")
            series_map = {
                "DIA oracle price": "dia_price",
                "Binance/combined price": "external_price",
                "StableSwap price": "stableswap_price",
                "StableSwap peg": "stableswap_peg",
                "StableSwap peg target": "stableswap_peg_target",
            }
            left_col, right_col = st.columns(2)
            left_label = left_col.selectbox(
                "Series A",
                list(series_map.keys()),
                index=0,
                key="spread_series_a",
            )
            right_options = [k for k in series_map.keys() if k != left_label]
            right_label = right_col.selectbox(
                "Series B",
                right_options,
                index=0,
                key="spread_series_b",
            )

            spread_df = series_df[["time", series_map[left_label], series_map[right_label]]].dropna()
            spread_df["spread"] = spread_df[series_map[left_label]] - spread_df[series_map[right_label]]

            spread_fig = go.Figure()
            spread_fig.add_trace(
                go.Scatter(
                    x=spread_df["time"],
                    y=spread_df["spread"],
                    name="Spread",
                    line=dict(color="#FF8C00", width=1.8),
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Spread: %{y:.6f}<extra></extra>",
                )
            )
            spread_fig.update_layout(
                height=420, template="plotly_dark", hovermode="x unified",
                margin=dict(t=40, b=40, l=60, r=20),
            )
            spread_fig.update_yaxes(title_text="Price Spread", tickformat=".6f")
            spread_fig.update_xaxes(title_text="Time (UTC)")
            st.plotly_chart(spread_fig, use_container_width=True)
        else:
            st.info("Run the simulation to view the spread chart.")

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

    if dia_range is not None:
        chart_start = pd.Timestamp(start_day, tz="UTC")
        chart_end = pd.Timestamp(end_day + timedelta(days=1), tz="UTC")
        if chart_start < dia_range[0] or chart_end > dia_range[1]:
            st.warning(
                "Selected range exceeds available DIA data. Available DIA range: "
                f"{dia_range[0].strftime('%Y-%m-%d %H:%M:%S UTC')} to "
                f"{dia_range[1].strftime('%Y-%m-%d %H:%M:%S UTC')}."
            )

    # ── Binance bias factor lives OUTSIDE the fragment because it affects the
    # price chart. Changing it redraws the chart and resets the simulation.
    binance_bias_slider = sidebar.slider(
        "Binance vs Kraken bias",
        min_value=-10.0, max_value=10.0,
        value=st.session_state.get("binance_bias_slider", 3.0),
        step=0.5,
        format=" ",
        help=(
            "0 is neutral (snaps within ±0.2). Nonzero values shift by ±1 internally "
            "to avoid the -1..1 zone; positive biases Binance, negative biases Kraken."
        ),
        key="binance_bias_slider",
    )

    label_cols = sidebar.columns(3)
    label_cols[0].markdown("Kraken", unsafe_allow_html=True)
    label_cols[1].markdown(
        f"<div style='text-align: center;'>{binance_bias_slider:+.1f}</div>",
        unsafe_allow_html=True,
    )
    label_cols[2].markdown("<div style='text-align: right;'>Binance</div>", unsafe_allow_html=True)

    if abs(binance_bias_slider) < 0.2:
        effective_slider = 0.0
    else:
        effective_slider = binance_bias_slider + (1.0 if binance_bias_slider > 0 else -1.0)

    if effective_slider == 0.0:
        binance_bias_factor = 1.0
    elif effective_slider > 0:
        binance_bias_factor = effective_slider
    else:
        binance_bias_factor = 1.0 / abs(effective_slider)

    combined_df = smooth_binance_with_kraken(binance_df, kraken_df, binance_bias_factor)
    st.session_state["combined_df"] = combined_df
    st.session_state["dia_df"] = dia_df
    if not combined_df.empty:
        st.session_state["simulation_start_time"] = combined_df["time"].min()
        st.session_state["simulation_end_time"] = combined_df["time"].max()

    if st.session_state.get("combined_df_hash") != binance_bias_slider:
        st.session_state["combined_df_hash"] = binance_bias_slider
        st.session_state["simulation_ran"] = False
        st.session_state["simulation_results"] = None

    if not combined_df.empty:
        dia_norm = _to_price_df(dia_df) if not dia_df.empty else pd.DataFrame(columns=["timestamp_ms", "time", "price"])

        toggle_cols = st.columns(4)
        show_binance  = toggle_cols[0].checkbox("Binance",  value=True, key="show_binance")
        show_kraken   = toggle_cols[1].checkbox("Kraken",   value=True, key="show_kraken")
        show_combined = toggle_cols[2].checkbox("Combined", value=True, key="show_combined")
        show_dia      = toggle_cols[3].checkbox("DIA",      value=True, key="show_dia")

        # Limit plot points to keep redraws responsive.
        max_plot_points = 1200
        plot_combined = _downsample_for_plot(combined_df, max_plot_points)
        if not dia_norm.empty:
            plot_start = combined_df["time"].min()
            plot_end = combined_df["time"].max()
            dia_norm = _pad_series_to_range(dia_norm, plot_start, plot_end)
        plot_dia = _downsample_for_plot(dia_norm, max_plot_points)

        fig = go.Figure()
        if show_binance:
            fig.add_trace(go.Scatter(
                x=plot_combined["time"], y=plot_combined["binance_price"],
                name="Binance", line=dict(color="#58D68D", width=1.2),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Binance</extra>",
            ))
        if show_kraken:
            fig.add_trace(go.Scatter(
                x=plot_combined["time"], y=plot_combined["kraken_price"],
                name="Kraken (interp)", line=dict(color="#F4D03F", width=1.2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>Kraken</extra>",
            ))
        if show_combined:
            fig.add_trace(go.Scatter(
                x=plot_combined["time"], y=plot_combined["external_price"],
                name="combined", line=dict(color="#E74C3C", width=1.6),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.6f}<extra>combined</extra>",
            ))
        if show_dia and not plot_dia.empty:
            fig.add_trace(go.Scatter(
                x=plot_dia["time"], y=plot_dia["price"],
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

    simulation_section()

# =============================================================================
# CLI DEMO
# =============================================================================
if __name__ == "__main__" and not st.runtime.exists():
    demo_day = date.today() - timedelta(days=1)
    binance_demo = get_prices_for_day("binance", demo_day)
    kraken_demo = get_prices_for_day("kraken", demo_day)
    demo = smooth_binance_with_kraken(binance_demo, kraken_demo, binance_bias_factor=3.0)
    print(demo.head(5).to_string(index=False))
