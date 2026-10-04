"""Market chart backend. Run: uvicorn main:app --reload   |  API docs: /docs"""
import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Market Chart API", description="Candles, indicators (SMA/EMA/RSI) and watchlist for the chart UI")
BASE_DIR = Path(__file__).parent
IST = 19800

PRESETS = {
    "crypto": [{"symbol": s, "name": n} for s, n in [("BTCUSDT", "Bitcoin"), ("ETHUSDT", "Ethereum"), ("SOLUSDT", "Solana"), ("BNBUSDT", "BNB"), ("XRPUSDT", "XRP")]],
    "stock": [{"symbol": s, "name": n} for s, n in [("^NSEI", "Nifty 50"), ("^NSEBANK", "Bank Nifty"), ("RELIANCE.NS", "Reliance"), ("TCS.NS", "TCS"), ("INFY.NS", "Infosys"), ("HDFCBANK.NS", "HDFC Bank"), ("SBIN.NS", "SBI")]],
}

CACHE, TTL = {}, 30


def cached(key, fn):
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    data = fn()
    CACHE[key] = (time.time(), data)
    return data


def normalize(source, s):
    s = s.strip().upper()
    if source == "stock" and not s.startswith("^") and "." not in s:
        s += ".NS"          # RELIANCE -> RELIANCE.NS
    if source == "crypto" and len(s) <= 5:
        s += "USDT"         # BTC -> BTCUSDT
    return s


# ---------- data sources ----------
BINANCE_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}
YF_MAP = {"1m": ("1m", "5d"), "5m": ("5m", "30d"), "15m": ("15m", "30d"), "1h": ("60m", "180d"), "1d": ("1d", "2y")}


def fetch_binance(symbol, interval, limit):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}, timeout=10)
    if r.status_code != 200:
        raise HTTPException(502, f"Binance error: {r.text[:200]}")
    return [{"time": k[0] // 1000, "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
             "close": float(k[4]), "volume": float(k[5])} for k in r.json()]


def fetch_yfinance(symbol, interval):
    if interval not in YF_MAP:
        raise HTTPException(400, f"Interval '{interval}' this value is not supported for stocks")
    yi, period = YF_MAP[interval]
    df = yf.Ticker(symbol).history(period=period, interval=yi)
    if df.empty:
        raise HTTPException(404, f"'{symbol}' no data found for interval ")
    return [{"time": int(ts.timestamp()), "open": round(float(r["Open"]), 2), "high": round(float(r["High"]), 2),
             "low": round(float(r["Low"]), 2), "close": round(float(r["Close"]), 2), "volume": float(r["Volume"])}
            for ts, r in df.iterrows()]


# ---------- indicators (pandas se) ----------
def clean(s):
    return [None if pd.isna(x) else float(x) for x in s]


def sma(v, n):   # pichli n candles ka simple average
    return clean(pd.Series(v).rolling(n).mean())


def ema(v, n):   # average, par recent price ko zyada weight
    return clean(pd.Series(v).ewm(span=n, adjust=False).mean())


def rsi(v, n=14):  # 0-100: 70+ zyada chadha, 30- zyada gira
    d = pd.Series(v).diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return clean(100 - 100 / (1 + gain / loss))


def series(times, vals):
    return [{"time": t, "value": round(x, 2)} for t, x in zip(times, vals) if x is not None]


def summary(source, rows):
    last = rows[-1]
    if source == "stock":   # pichle session ke close se compare
        day = lambda c: (c["time"] + IST) // 86400
        d = day(last)
        today, prev = [c for c in rows if day(c) == d], [c for c in rows if day(c) < d]
        base = prev[-1]["close"] if prev else today[0]["open"]
    else:                   # crypto: 24 ghante pehle se compare
        cut = last["time"] - 86400
        today, old = [c for c in rows if c["time"] > cut], [c for c in rows if c["time"] <= cut]
        base = old[-1]["close"] if old else rows[0]["open"]
    ch = last["close"] - base
    return {"last": last["close"], "change": round(ch, 2), "change_pct": round(ch / base * 100, 2),
            "high": max(c["high"] for c in today), "low": min(c["low"] for c in today),
            "volume": sum(c["volume"] for c in today)}


# ---------- endpoints ----------
@app.get("/api/symbols", tags=["market"])
def symbols():
    """Preset symbols for each market."""
    return PRESETS


@app.get("/api/candles", tags=["market"])
def candles(source: str = Query("crypto", pattern="^(crypto|stock)$"), symbol: str = "BTCUSDT",
            interval: str = "1h", limit: int = Query(300, ge=10, le=1000)):
    """OHLCV candles + SMA20 / EMA50 / RSI14 + 1D summary."""
    symbol = normalize(source, symbol)
    if source == "crypto":
        if interval not in BINANCE_INTERVALS:
            raise HTTPException(400, "Invalid interval")
        rows = cached(("c", symbol, interval, limit), lambda: fetch_binance(symbol, interval, limit + 60))
    else:
        rows = cached(("s", symbol, interval), lambda: fetch_yfinance(symbol, interval))
    closes, times = [c["close"] for c in rows], [c["time"] for c in rows]
    cut = max(len(rows) - limit, 0)
    ind = {"sma20": series(times[cut:], sma(closes, 20)[cut:]), "ema50": series(times[cut:], ema(closes, 50)[cut:]),
           "rsi14": series(times[cut:], rsi(closes)[cut:])}
    rows = rows[cut:]
    return {"source": source, "symbol": symbol, "interval": interval, "candles": rows,
            "indicators": ind, "summary": summary(source, rows)}


@app.get("/api/watchlist", tags=["market"])
def watchlist(source: str = Query("crypto", pattern="^(crypto|stock)$")):
    """Last price and 1D change for the preset symbols."""
    items = PRESETS[source]

    def build():
        if source == "crypto":
            r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                             params={"symbols": json.dumps([i["symbol"] for i in items], separators=(",", ":"))}, timeout=10)
            if r.status_code != 200:
                raise HTTPException(502, "Binance watchlist error")
            m = {x["symbol"]: x for x in r.json()}
            return [{**i, "last": float(m[i["symbol"]]["lastPrice"]), "change_pct": float(m[i["symbol"]]["priceChangePercent"])}
                    for i in items if i["symbol"] in m]

        def one(i):
            try:
                h = yf.Ticker(i["symbol"]).history(period="5d", interval="1d")["Close"].dropna()
                return {**i, "last": round(float(h.iloc[-1]), 2), "change_pct": round(float(h.iloc[-1] / h.iloc[-2] - 1) * 100, 2)}
            except Exception:
                return None
        with ThreadPoolExecutor(6) as ex:
            return [x for x in ex.map(one, items) if x]

    return cached(("wl", source), build)


if (BASE_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")