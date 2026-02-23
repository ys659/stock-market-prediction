from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
from fastapi import APIRouter, HTTPException
from fastapi.params import Depends
from sqlalchemy.orm import Session

from app.database import database, crud, models
from ..DTO.marketenums import MarketEnum
from ..DTO.schemas import StockData, HistoricalData, StockSearch

router = APIRouter()

# Reuse one session for all Yahoo calls (helps avoid 429 rate limiting)
YF_SESSION = curl_requests.Session(impersonate="chrome")


@router.get("/stocks")
async def read_items():
    return {"message": "Endpoint works!"}


@router.get("/stocks/{symbol}", response_model=StockData)
async def get_stock_data(symbol: str, db: Session = Depends(database.get_db)):
    """Get (and cache) stock metadata.

    NOTE: Yahoo frequently rate-limits yfinance.
    Using curl_cffi session impersonation helps.
    """
    symbol = symbol.upper().strip()

    # 1) DB cache first
    db_stock = crud.get_stock_by_ticker(db, symbol)
    if db_stock:
        return db_stock

    try:
        t = yf.Ticker(symbol, session=YF_SESSION)

        # 2) Lightweight existence check (much lighter than .info)
        hist = t.history(period="1d")
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail="Stock not found")

        # 3) fast_info is usually cheaper than get_info/info
        fi = getattr(t, "fast_info", None) or {}

        # 4) Best-effort "pretty" metadata (may still rate-limit)
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}

        name = info.get("shortName") or info.get("longName") or symbol
        exchange = (
            info.get("exchange")
            or info.get("fullExchangeName")
            or fi.get("exchange")
        )
        currency = info.get("currency") or fi.get("currency")

        stock_payload = {
            "ticker": symbol,
            "name": name,
            "market": MarketEnum.STOCKS.value,
            "locale": info.get("country") or info.get("region"),
            "primary_exchange": exchange,
            "type": info.get("quoteType"),
            "active": True,
            "currency_name": currency,
            "cik": str(info.get("cik")) if info.get("cik") is not None else None,
            "composite_figi": info.get("compositeFigi"),
            "share_class_figi": info.get("shareClassFigi"),
        }

        db_stock = crud.create_stock(db, stock_payload)
        return db_stock

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {e}")


@router.get("/stocks/{symbol}/historical", response_model=HistoricalData)
async def get_historical_data(
    symbol: str,
    date_from: str,
    date_to: str,
    multiplier: int,
    timespan: str,
    db: Session = Depends(database.get_db),
):
    symbol = symbol.upper().strip()

    try:
        await get_stock_data(symbol, db)
    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Stock not found")
        raise

    try:
        date_from_obj = datetime.strptime(date_from, "%Y-%m-%d")
        date_to_obj = datetime.strptime(date_to, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date_from/date_to must be YYYY-MM-DD")

    if date_from_obj > date_to_obj:
        raise HTTPException(status_code=400, detail="date_from must be <= date_to")

    # Keep the same signature as the tutorial, but this implementation caches DAILY bars only.
    if str(timespan).lower() not in {"day", "daily"} or int(multiplier) != 1:
        raise HTTPException(
            status_code=400,
            detail="This endpoint currently supports multiplier=1 and timespan=day only.",
        )

    # Try DB first
    results = []
    current_date = date_from_obj
    all_data_exists = True

    while current_date <= date_to_obj:
        db_data = (
            db.query(models.HistoricalStockData)
            .filter(
                models.HistoricalStockData.symbol == symbol,
                models.HistoricalStockData.timestamp == current_date,
            )
            .first()
        )

        if not db_data:
            all_data_exists = False
            break
        else:
            results.append(
                {
                    "o": db_data.open,
                    "h": db_data.high,
                    "l": db_data.low,
                    "c": db_data.close,
                    "v": db_data.volume,
                    "t": int(db_data.timestamp.timestamp() * 1000),
                }
            )
        current_date += timedelta(days=1)

    if all_data_exists:
        return {
            "ticker": symbol,
            "queryCount": len(results),
            "resultsCount": len(results),
            "adjusted": True,
            "results": results,
            "status": "OK",
            "request_id": "db_fetch",
            "count": len(results),
        }

    # Fetch missing data from yfinance and backfill DB
    try:
        ticker = yf.Ticker(symbol, session=YF_SESSION)

        # yfinance end is effectively exclusive for many calls; add 1 day to include date_to.
        yf_start = date_from_obj.date().isoformat()
        yf_end = (date_to_obj.date() + timedelta(days=1)).isoformat()

        hist = ticker.history(
            start=yf_start,
            end=yf_end,
            interval="1d",
            auto_adjust=True,
        )

        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail="No historical data found")

        hist = hist.reset_index()

        def to_midnight_dt(x) -> datetime:
            ts = pd.to_datetime(x)
            d = ts.date()
            return datetime(d.year, d.month, d.day)

        out_results = []
        for _, row in hist.iterrows():
            ts = to_midnight_dt(row["Date"])
            if ts < date_from_obj or ts > date_to_obj:
                continue

            # Save if missing
            exists = (
                db.query(models.HistoricalStockData)
                .filter(
                    models.HistoricalStockData.symbol == symbol,
                    models.HistoricalStockData.timestamp == ts,
                )
                .first()
            )
            if not exists:
                crud.create_historical_data(
                    db,
                    {
                        "symbol": symbol,
                        "timestamp": ts,
                        "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                        "high": float(row["High"]) if pd.notna(row["High"]) else None,
                        "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                        "close": float(row["Close"]) if pd.notna(row["Close"]) else None,
                        "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
                    },
                )

            out_results.append(
                {
                    "o": float(row["Open"]) if pd.notna(row["Open"]) else None,
                    "h": float(row["High"]) if pd.notna(row["High"]) else None,
                    "l": float(row["Low"]) if pd.notna(row["Low"]) else None,
                    "c": float(row["Close"]) if pd.notna(row["Close"]) else None,
                    "v": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
                    "t": int(ts.timestamp() * 1000),
                }
            )

        return {
            "ticker": symbol,
            "queryCount": len(out_results),
            "resultsCount": len(out_results),
            "adjusted": True,
            "results": out_results,
            "status": "OK",
            "request_id": "yfinance_fetch",
            "count": len(out_results),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {e}")


@router.get("/stocks/search/{query}", response_model=StockSearch)
async def search_stocks(query: str):
    """Best-effort ticker search.

    yfinance doesn't guarantee stable search behavior. If Search isn't available,
    we return 501.
    """
    q = query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not hasattr(yf, "Search"):
        raise HTTPException(
            status_code=501,
            detail="Ticker search not supported by this yfinance version. Upgrade yfinance or implement your own search.",
        )

    try:
        # If your yfinance version supports it, pass session to Search too.
        # Some versions accept session=..., some don't — so we try and fallback.
        try:
            s = yf.Search(q, session=YF_SESSION)
        except TypeError:
            s = yf.Search(q)

        quotes = getattr(s, "quotes", None) or []

        results = []
        for item in quotes:
            symbol = (item.get("symbol") or "").upper()
            if not symbol:
                continue
            results.append(
                {
                    "ticker": symbol,
                    "name": item.get("shortname") or item.get("longname") or item.get("name"),
                    "market": MarketEnum.STOCKS.value,
                    "primary_exchange": item.get("exchange"),
                    "type": item.get("quoteType") or item.get("typeDisp"),
                    "active": True,
                }
            )

        return {
            "results": results,
            "count": len(results),
            "status": "OK",
            "request_id": "yfinance_search",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch search results: {e}")