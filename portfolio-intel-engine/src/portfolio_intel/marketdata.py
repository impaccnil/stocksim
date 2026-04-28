from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd
import requests


Timeframe = Literal["1d", "1h", "15m", "5m", "1m"]


@dataclass(frozen=True)
class OHLCVRequest:
    ticker: str
    start: datetime
    end: datetime
    timeframe: Timeframe = "1d"


class OHLCVProvider(Protocol):
    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume
        """


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}. Got: {list(df.columns)}")
    out = df.rename(columns={cols[c]: c for c in need})[need].copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    return out


class DiskOHLCVCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, timeframe: Timeframe) -> Path:
        safe = ticker.upper().replace("/", "_")
        return self.root / f"{safe}_{timeframe}.parquet"

    def read(self, ticker: str, timeframe: Timeframe) -> pd.DataFrame | None:
        p = self._path(ticker, timeframe)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        return _ensure_ohlcv_schema(df)

    def write(self, ticker: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        p = self._path(ticker, timeframe)
        out = _ensure_ohlcv_schema(df)
        out.to_parquet(p)


class CachedOHLCVProvider:
    def __init__(self, inner: OHLCVProvider, cache: DiskOHLCVCache) -> None:
        self.inner = inner
        self.cache = cache

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        cached = self.cache.read(req.ticker, req.timeframe)
        if cached is None:
            df = self.inner.get_ohlcv(req)
            self.cache.write(req.ticker, req.timeframe, df)
            return df.loc[_utc(req.start) : _utc(req.end)]

        # If cache doesn't cover range, we still refetch full range (simple, reliable).
        if cached.index.min() > _utc(req.start) or cached.index.max() < _utc(req.end):
            df = self.inner.get_ohlcv(req)
            self.cache.write(req.ticker, req.timeframe, df)
            return df.loc[_utc(req.start) : _utc(req.end)]

        return cached.loc[_utc(req.start) : _utc(req.end)]


class YahooFinanceProvider:
    """
    Free-ish data source via yfinance. Suitable for prototyping.
    """

    def __init__(self) -> None:
        import yfinance as yf  # deferred import

        self.yf = yf

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        interval_map: dict[Timeframe, str] = {
            "1d": "1d",
            "1h": "60m",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m",
        }
        interval = interval_map[req.timeframe]
        df = self.yf.download(
            req.ticker,
            start=_utc(req.start).isoformat(),
            end=_utc(req.end).isoformat(),
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if df is None or df.empty:
            raise ValueError(f"No Yahoo OHLCV returned for {req.ticker}.")
        df = df.rename(columns=str.lower)
        # yfinance sometimes returns multiindex columns when many tickers; we request one ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.lower)
        return _ensure_ohlcv_schema(df)


class AlpacaMarketDataProvider:
    """
    Alpaca market data v2 (data only). Requires env vars:
      ALPACA_KEY_ID, ALPACA_SECRET_KEY
    Optional:
      ALPACA_DATA_BASE (default https://data.alpaca.markets)
    """

    def __init__(self) -> None:
        self.key = os.environ.get("ALPACA_KEY_ID", "").strip()
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
        self.base = os.environ.get("ALPACA_DATA_BASE", "https://data.alpaca.markets").strip().rstrip("/")
        if not self.key or not self.secret:
            raise ValueError("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY environment variables.")

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        tf_map: dict[Timeframe, str] = {"1d": "1Day", "1h": "1Hour", "15m": "15Min", "5m": "5Min", "1m": "1Min"}
        url = f"{self.base}/v2/stocks/{req.ticker.upper()}/bars"
        params = {
            "timeframe": tf_map[req.timeframe],
            "start": _utc(req.start).isoformat(),
            "end": _utc(req.end).isoformat(),
            "adjustment": "raw",
            "limit": 10000,
        }
        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()
        bars = payload.get("bars") or []
        if not bars:
            raise ValueError(f"No Alpaca bars returned for {req.ticker}.")
        df = pd.DataFrame(bars)
        # Alpaca fields: t,o,h,l,c,v
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        return _ensure_ohlcv_schema(df)


class PolygonProvider:
    """
    Polygon aggregates endpoint. Requires env var:
      POLYGON_API_KEY
    """

    def __init__(self) -> None:
        self.key = os.environ.get("POLYGON_API_KEY", "").strip()
        self.base = os.environ.get("POLYGON_BASE", "https://api.polygon.io").strip().rstrip("/")
        if not self.key:
            raise ValueError("Missing POLYGON_API_KEY environment variable.")

    def get_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        # Map timeframe to Polygon multiplier/timespan
        tf = req.timeframe
        mult, span = {
            "1d": (1, "day"),
            "1h": (1, "hour"),
            "15m": (15, "minute"),
            "5m": (5, "minute"),
            "1m": (1, "minute"),
        }[tf]
        start_ms = int(_utc(req.start).timestamp() * 1000)
        end_ms = int(_utc(req.end).timestamp() * 1000)
        url = f"{self.base}/v2/aggs/ticker/{req.ticker.upper()}/range/{mult}/{span}/{start_ms}/{end_ms}"
        params = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": self.key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"No Polygon aggs returned for {req.ticker}.")
        df = pd.DataFrame(results)
        # Polygon fields: t,o,h,l,c,v
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        return _ensure_ohlcv_schema(df)


def provider_from_env() -> OHLCVProvider:
    """
    Select a provider based on env var MARKET_DATA_PROVIDER:
      yahoo (default), alpaca, polygon
    """
    which = os.environ.get("MARKET_DATA_PROVIDER", "yahoo").strip().lower()
    if which == "alpaca":
        return AlpacaMarketDataProvider()
    if which == "polygon":
        return PolygonProvider()
    return YahooFinanceProvider()


def default_cache_dir() -> Path:
    return Path("data") / "ohlcv_cache"

