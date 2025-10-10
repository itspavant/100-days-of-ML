import os
import re
import time
import math
import hashlib
from typing import Iterable, Optional, Tuple, Dict, Any

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import numpy as np
from tqdm import tqdm


def _ensure_directory(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _build_requests_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/117.0 Safari/537.36"
        )
    })
    return session


def _infer_extension_from_headers(content_type: Optional[str]) -> str:
    if not content_type:
        return ".jpg"
    content_type = content_type.lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "bmp" in content_type:
        return ".bmp"
    if "gif" in content_type:
        return ".gif"
    return ".jpg"


def _safe_filename(sample_id: Any, url: str, preferred_ext: Optional[str] = None) -> str:
    hasher = hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:12]
    sid = str(sample_id)
    ext = preferred_ext if preferred_ext else ".jpg"
    return f"{sid}_{hasher}{ext}"


def download_image(
    url: str,
    dest_dir: str,
    sample_id: Any,
    timeout: int = 15,
    max_retries: int = 5,
    sleep_on_error: float = 0.5,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """
    Download a single image with retries. Returns saved file path or None.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    _ensure_directory(dest_dir)

    owns_session = False
    if session is None:
        session = _build_requests_session(total_retries=max_retries)
        owns_session = True

    try:
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = session.get(url, timeout=timeout, stream=True)
                if resp.status_code != 200:
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(sleep_on_error * (attempt + 1))
                    continue

                content_type = resp.headers.get("Content-Type")
                ext = _infer_extension_from_headers(content_type)
                filename = _safe_filename(sample_id=sample_id, url=url, preferred_ext=ext)
                out_path = os.path.join(dest_dir, filename)

                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Basic sanity check on file size
                if os.path.getsize(out_path) < 1024:  # < 1KB likely bad
                    os.remove(out_path)
                    last_exc = RuntimeError("Downloaded file too small; likely throttled or invalid")
                    time.sleep(sleep_on_error * (attempt + 1))
                    continue

                return out_path
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(sleep_on_error * (attempt + 1))
                continue
        if last_exc:
            raise last_exc
    finally:
        if owns_session and session is not None:
            session.close()
    return None


def download_images(
    df: pd.DataFrame,
    image_link_col: str = "image_link",
    sample_id_col: str = "sample_id",
    dest_dir: str = "images",
    timeout: int = 15,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Download images for all rows in df and return a new DataFrame with
    an additional column `image_path` containing the local path when available.
    """
    if image_link_col not in df.columns or sample_id_col not in df.columns:
        raise KeyError(
            f"DataFrame must contain columns '{image_link_col}' and '{sample_id_col}'"
        )

    _ensure_directory(dest_dir)
    session = _build_requests_session(total_retries=max_retries)
    local_paths: list[Optional[str]] = []

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
            url = row[image_link_col]
            sample_id = row[sample_id_col]
            path = download_image(
                url=url,
                dest_dir=dest_dir,
                sample_id=sample_id,
                timeout=timeout,
                max_retries=max_retries,
                session=session,
            )
            local_paths.append(path)
    finally:
        session.close()

    result = df.copy()
    result["image_path"] = local_paths
    return result


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE) in percentage [0, 200].
    """
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    # Guard against negatives from models (we don't expect negative prices)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero: if both true and pred are zero, contribution is 0
    mask = denominator != 0
    ratio = np.zeros_like(denominator)
    ratio[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(np.mean(ratio) * 100.0)


# --------- Lightweight text feature helpers ---------

_TOKEN_PATTERN = re.compile(r"\b\d{1,5}(?:[.,]\d{1,2})?\b")


def extract_item_pack_quantity(text: Optional[str]) -> float:
    """
    Heuristic extraction of an Item Pack Quantity (IPQ) from unstructured text.
    Returns 1.0 if no clear quantity is found.
    """
    if not isinstance(text, str) or not text:
        return 1.0

    lowered = text.lower()

    # Common explicit patterns
    patterns = [
        r"pack\s*of\s*(\d+)",
        r"(\d+)\s*pack\b",
        r"(\d+)\s*pcs\b",
        r"(\d+)\s*pieces\b",
        r"(\d+)\s*count\b",
        r"(\d+)\s*ct\b",
        r"(\d+)\s*x\s*(\d+)",  # e.g., 12 x 2
        r"(\d+)x(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            # If two groups, multiply, else take the number
            if len(m.groups()) == 2:
                try:
                    return float(int(m.group(1)) * int(m.group(2)))
                except Exception:  # noqa: BLE001
                    continue
            try:
                return float(int(m.group(1)))
            except Exception:  # noqa: BLE001
                continue

    # Fallback: take the largest small integer token as a guess
    candidates = []
    for tok in _TOKEN_PATTERN.findall(lowered):
        try:
            candidates.append(int(tok.replace(",", "").split(".")[0]))
        except Exception:  # noqa: BLE001
            continue
    if candidates:
        return float(max(candidates))

    return 1.0


def add_basic_text_features(df: pd.DataFrame, text_col: str = "catalog_content") -> pd.DataFrame:
    """
    Add simple numeric features derived from the text column:
    - text_len: number of characters
    - num_digits: count of digits
    - ipq: heuristic Item Pack Quantity
    """
    out = df.copy()
    series = out[text_col].fillna("").astype(str)
    out["text_len"] = series.str.len()
    out["num_digits"] = series.str.count(r"\d")
    out["ipq"] = series.apply(extract_item_pack_quantity)
    return out


__all__ = [
    "download_image",
    "download_images",
    "smape",
    "extract_item_pack_quantity",
    "add_basic_text_features",
]
