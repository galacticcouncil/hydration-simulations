import mimetypes
import os
from typing import Callable
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st


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
# If none of the above are set, cloud storage is silently disabled.
# =============================================================================


def _secrets_paths() -> list[Path]:
    paths = []
    env_path = os.environ.get("S3_SECRETS_PATH")
    if env_path:
        paths.append(Path(env_path))
    env_dir = os.environ.get("STREAMLIT_SECRETS_DIR")
    if env_dir:
        paths.append(Path(env_dir) / "secrets.toml")

    repo_root = Path(__file__).resolve().parents[2]
    paths.append(repo_root / ".streamlit" / "secrets.toml")

    cwd_path = Path.cwd() / ".streamlit" / "secrets.toml"
    if cwd_path not in paths:
        paths.append(cwd_path)

    return paths


def _load_secrets_file() -> dict:
    try:
        import tomllib
    except Exception:
        try:
            import tomli as tomllib
        except Exception:
            print("[cloud] tomllib/tomli not available; cannot read secrets files.")
            return {}

    for secrets_path in _secrets_paths():
        if not secrets_path.exists():
            continue
        try:
            with secrets_path.open("rb") as handle:
                data = tomllib.load(handle)
            if not isinstance(data, dict):
                continue
            print(f"[cloud] Loaded secrets from {secrets_path}")
            return data
        except Exception as exc:
            print(f"[cloud] Failed to read secrets file {secrets_path}: {exc}")
            return {}

    missing_paths = ", ".join(str(p) for p in _secrets_paths())
    print(f"[cloud] No secrets file found in configured paths: {missing_paths}")
    return {}


def get_s3_config() -> dict | None:
    """
    Return S3 config dict from st.secrets or environment variables,
    or None if cloud storage is not configured.
    """
    cfg = {}
    bucket = None
    try:
        cfg = st.secrets.get("s3", {})
        bucket = cfg.get("bucket")
    except Exception as exc:
        print(f"[cloud] st.secrets unavailable: {exc}")

    if not bucket:
        bucket = os.environ.get("S3_BUCKET")

    if not bucket:
        file_cfg = _load_secrets_file().get("s3", {})
        if file_cfg:
            cfg = {**file_cfg, **cfg}
            bucket = cfg.get("bucket")

    if not bucket:
        print("[cloud] S3 bucket not configured; skipping cloud cache.")
        return None

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
        "endpoint_url": _get("endpoint_url", "S3_ENDPOINT_URL"),
    }


@st.cache_resource(show_spinner=False)
def get_s3_client():
    """Return a boto3 S3 client (cached for the session), or (None, None)."""
    cfg = get_s3_config()
    if cfg is None:
        print("[cloud] S3 config missing; client not created.")
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


def s3_key(exchange: str, day: date, cfg: dict) -> str:
    prefix = cfg["prefix"].rstrip("/")
    key = f"{exchange}/{day.isoformat()}.csv"
    return f"{prefix}/{key}" if prefix else key


def s3_join(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part)


def _resolve_bucket(cfg: dict, bucket_override: str | None) -> str:
    return bucket_override or cfg["bucket"]


def download_file_from_s3(key: str, dest_path: Path, bucket: str | None = None) -> bool:
    client, cfg = get_s3_client()
    if client is None:
        return False

    bucket_name = _resolve_bucket(cfg, bucket)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = client.get_object(Bucket=bucket_name, Key=key)
        body = response["Body"].read()
        dest_path.write_bytes(body)
        print(f"[cloud] Downloaded s3://{bucket_name}/{key} -> {dest_path}")
        return True
    except client.exceptions.NoSuchKey:
        return False
    except Exception as exc:
        print(f"[cloud] S3 download failed for {key}: {exc}")
        return False


def upload_file_to_s3(path: Path, key: str, content_type: str | None = None, bucket: str | None = None) -> bool:
    client, cfg = get_s3_client()
    if client is None:
        return False

    if not path.exists():
        return False

    bucket_name = _resolve_bucket(cfg, bucket)
    try:
        body = path.read_bytes()
        guessed_type, _ = mimetypes.guess_type(str(path))
        final_type = content_type or guessed_type or "application/octet-stream"
        client.put_object(Bucket=bucket_name, Key=key, Body=body, ContentType=final_type)
        print(f"[cloud] Uploaded {path} -> s3://{bucket_name}/{key}")
        return True
    except Exception as exc:
        print(f"[cloud] S3 upload failed for {key}: {exc}")
        return False


def list_s3_keys(prefix: str, bucket: str | None = None) -> list[str]:
    client, cfg = get_s3_client()
    if client is None:
        return []

    bucket_name = _resolve_bucket(cfg, bucket)
    keys = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
    except Exception as exc:
        print(f"[cloud] S3 list failed for prefix {prefix}: {exc}")
    return keys


def sync_dir_from_s3(
    local_dir: Path,
    prefix: str,
    bucket: str | None = None,
    progress_cb: Callable[[int, int, int], None] | None = None,
) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    keys = list_s3_keys(prefix, bucket=bucket)
    total = len(keys)
    if progress_cb:
        progress_cb(0, total, downloaded)
    for idx, key in enumerate(keys, start=1):
        rel = key[len(prefix):].lstrip("/")
        if not rel:
            if progress_cb:
                progress_cb(idx, total, downloaded)
            continue
        dest = local_dir / rel
        if dest.exists():
            if progress_cb:
                progress_cb(idx, total, downloaded)
            continue
        if download_file_from_s3(key, dest, bucket=bucket):
            downloaded += 1
        if progress_cb:
            progress_cb(idx, total, downloaded)
    return downloaded


def upload_dir_to_s3(local_dir: Path, prefix: str, bucket: str | None = None) -> int:
    if not local_dir.exists():
        return 0

    uploaded = 0
    for path in local_dir.rglob("*"):
        if path.is_dir():
            continue
        key = s3_join(prefix, path.relative_to(local_dir).as_posix())
        if upload_file_to_s3(path, key, bucket=bucket):
            uploaded += 1
    return uploaded


def download_from_s3(exchange: str, day: date) -> pd.DataFrame | None:
    """
    Try to download a cached CSV from S3. Returns a DataFrame on success,
    None if the object doesn't exist or cloud storage is unconfigured.
    """
    client, cfg = get_s3_client()
    if client is None:
        return None

    key = s3_key(exchange, day, cfg)
    try:
        response = client.get_object(Bucket=cfg["bucket"], Key=key)
        body = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(body))
        print(f"[cloud] Downloaded {exchange}/{day} from s3://{cfg['bucket']}/{key}")
        return df
    except client.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        print(f"[cloud] S3 download failed for {key}: {exc}")
        return None


def upload_to_s3(df: pd.DataFrame, exchange: str, day: date) -> None:
    """Upload a price DataFrame as CSV to S3. Silently skips on any error."""
    client, cfg = get_s3_client()
    if client is None:
        return

    key = s3_key(exchange, day, cfg)
    try:
        body = df.to_csv(index=False).encode("utf-8")
        client.put_object(Bucket=cfg["bucket"], Key=key, Body=body, ContentType="text/csv")
        print(f"[cloud] Uploaded {exchange}/{day} to s3://{cfg['bucket']}/{key}")
    except Exception as exc:
        print(f"[cloud] S3 upload failed for {key}: {exc}")

