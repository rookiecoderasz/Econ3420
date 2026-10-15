#!/usr/bin/env python3
"""Download adjusted daily prices and the matching bill-yield proxy for A1.

Run from the assignment folder, for example:
  python tools/download_data.py --market US --stocks AAPL KO JPM XOM NVDA --benchmark SPY
  python tools/download_data.py --market HK --stocks 0700.HK 0005.HK 0939.HK 0388.HK 0016.HK --benchmark 2800.HK

This tool downloads inputs only. It performs no regressions or optimization.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
HKMA_URL = (
    "https://api.hkma.gov.hk/public/market-data-and-statistics/"
    "monthly-statistical-bulletin/efbn/efbn-yield-daily"
)
MARKETS = {"US": "USD", "HK": "HKD"}


class DataError(Exception):
    """A source or input is unusable; never replace it with invented data."""


def request_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ECON3420-A1-data-tool/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def clean_rates(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    # FRED's '.', and HKMA's nulls, are missing observations, not zero yields.
    raw = frame["annual_yield_percent"]
    known_missing = raw.isna() | raw.astype(str).str.strip().isin(["", ".", "nan", "None"])
    values = pd.to_numeric(raw, errors="coerce")
    if (values.isna() & ~known_missing).any():
        raise DataError("The bill source contains a non-numeric yield.")
    frame["annual_yield_percent"] = values
    frame = frame.dropna(subset=["annual_yield_percent"])
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] < end)]
    if frame["date"].duplicated().any():
        raise DataError("The bill source contains duplicate observation dates.")
    if frame.empty or not frame["annual_yield_percent"].map(lambda x: float('-inf') < x < float('inf')).all():
        raise DataError("No usable finite bill-yield observations were returned.")
    return frame.sort_values("date").reset_index(drop=True)


def fetch_fred(start: str, end: str) -> tuple[pd.DataFrame, dict, dict[str, bytes]]:
    url = FRED_URL + "?" + urlencode({
        "id": "DGS3MO", "cosd": start,
        "coed": (date.fromisoformat(end) - timedelta(days=1)).isoformat(),
    })
    raw = request_bytes(url)
    frame = pd.read_csv(io.BytesIO(raw), dtype=str)
    if "DGS3MO" not in frame or len(frame.columns) != 2:
        raise DataError("FRED returned an unexpected CSV format.")
    frame.columns = ["date", "annual_yield_percent"]
    return clean_rates(frame, start, end), {
        "name": "FRED DGS3MO: 3-month Treasury constant-maturity yield, investment basis",
        "source_page": "https://fred.stlouisfed.org/series/DGS3MO",
        "request_urls": [url], "frequency": "daily observations",
        "currency": "USD", "units": "percent per annum", "date_kind": "actual_observation_date",
    }, {"fred_DGS3MO.csv": raw}


def fetch_hkma(start: str, end: str) -> tuple[pd.DataFrame, dict, dict[str, bytes]]:
    records, urls, cached = [], [], {}
    offset, page_size = 0, 1000
    while True:
        url = HKMA_URL + "?" + urlencode({
            "choose": "end_of_day", "from": start,
            "to": (date.fromisoformat(end) - timedelta(days=1)).isoformat(),
            "fields": "end_of_day,efb_91d", "sortby": "end_of_day",
            "sortorder": "asc", "pagesize": page_size, "offset": offset,
        })
        raw = request_bytes(url)
        payload = json.loads(raw)
        header, result = payload.get("header", {}), payload.get("result", {})
        if header.get("success") is False or not isinstance(result.get("records"), list):
            raise DataError("HKMA returned an error or an unexpected JSON format.")
        page = result["records"]
        urls.append(url)
        cached[f"hkma_91day_offset_{offset}.json"] = raw
        records.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
        if offset > 100_000:
            raise DataError("HKMA pagination did not finish; no data files were replaced.")
    frame = pd.DataFrame(records)
    if not {"end_of_day", "efb_91d"}.issubset(frame.columns):
        raise DataError("HKMA returned no usable 91-day bill records.")
    frame = frame.rename(columns={"end_of_day": "date", "efb_91d": "annual_yield_percent"})
    return clean_rates(frame[["date", "annual_yield_percent"]], start, end), {
        "name": "HKMA 91-day Exchange Fund Bill indicative yield",
        "source_page": "https://apidocs.hkma.gov.hk/documentation/market-data-and-statistics/monthly-statistical-bulletin/efbn/efbn-yield-daily/",
        "request_urls": urls, "frequency": "daily observations",
        "currency": "HKD", "units": "percent per annum", "date_kind": "actual_observation_date",
        "date_field": "end_of_day", "yield_field": "efb_91d",
    }, cached


def load_supplied_rates(path: Path, currency: str, date_kind: str,
                        start: str, end: str) -> tuple[pd.DataFrame, dict, dict[str, bytes]]:
    """An explicitly selected cash input; never an automatic network fallback."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise DataError(f"The supplied risk-free CSV does not exist: {path}")
    raw = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), dtype=str)
    if list(frame.columns) != ["date", "annual_yield_percent"]:
        raise DataError("Supplied cash CSV must have exactly date,annual_yield_percent columns; quotes are percent per annum.")
    declaration_basis = "User explicitly declared the supplied CSV currency with --riskfree-currency."
    original_metadata = None
    metadata_path = path.with_name("metadata.json")
    if metadata_path.is_file():
        original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        known_currency = original_metadata.get("riskfree", {}).get("currency", original_metadata.get("currency"))
        if known_currency and known_currency != currency:
            raise DataError(f"Supplied cash provenance declares {known_currency}, but --riskfree-currency declares {currency}.")
        known_units = original_metadata.get("riskfree", {}).get("units")
        if known_units and ("percent" not in known_units.lower() or "monthly" in known_units.lower()):
            raise DataError("Supplied cash provenance contradicts the required annual-percent quote units.")
        known_date_kind = original_metadata.get("riskfree", {}).get("date_kind")
        requested_kind = {"actual-observation": "actual_observation_date", "source-period": "source_period_label"}[date_kind]
        if known_date_kind and known_date_kind != requested_kind:
            raise DataError(f"Supplied cash provenance date kind is {known_date_kind}; choose the matching --riskfree-date-kind.")
        declaration_basis += " Adjacent metadata.json was checked when present."
    info = {
        "name": "Explicitly supplied cash-rate CSV",
        "input_mode": "supplied_csv", "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "currency": currency, "declared_currency": currency,
        "currency_confirmation_basis": declaration_basis,
        "date_kind": {"actual-observation": "actual_observation_date", "source-period": "source_period_label"}[date_kind],
        "frequency": "source-period snapshots" if date_kind == "source-period" else "actual dated observations",
        "units": "percent per annum (declared by the annual_yield_percent schema)",
        "original_provenance": original_metadata,
        "source_cache_file": "supplied_riskfree.csv",
    }
    return clean_rates(frame, start, end), info, {"supplied_riskfree.csv": raw}


def fetch_prices(symbols: list[str], currency: str, start: str, end: str,
                 cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    # Keep yfinance's cookie/time-zone cache alongside this download, not elsewhere.
    yf.set_tz_cache_location(str(cache_dir))
    series, details = [], {}
    for symbol in symbols:
        print(f"Downloading adjusted daily prices: {symbol}", flush=True)
        ticker = yf.Ticker(symbol)
        history = ticker.history(start=start, end=end, interval="1d", auto_adjust=True,
                                 actions=False, repair=False, keepna=True,
                                 timeout=30)
        if history is None or history.empty or "Close" not in history:
            raise DataError(f"Yahoo returned no adjusted closes for {symbol}.")
        metadata = ticker.get_history_metadata()
        reported_currency = metadata.get("currency")
        if reported_currency != currency:
            raise DataError(f"{symbol}: source currency is {reported_currency!r}; expected {currency}.")
        close = history["Close"].copy()
        # Preserve the exchange-local calendar date of daily bars.
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        close = close.loc[(close.index >= start) & (close.index < end)]
        if close.index.duplicated().any():
            raise DataError(f"Yahoo returned duplicate daily dates for {symbol}.")
        if close.dropna().empty or not close.dropna().map(lambda x: 0 < x < float('inf')).all():
            raise DataError(f"{symbol}: missing, nonpositive, or nonfinite prices.")
        close.name = symbol
        series.append(close)
        details[symbol] = {
            "source_page": f"https://finance.yahoo.com/quote/{symbol}/history/",
            "currency": reported_currency, "exchange": metadata.get("fullExchangeName", metadata.get("exchangeName")),
            "instrument_type": metadata.get("instrumentType"),
            "first_date": close.first_valid_index().date().isoformat(),
            "last_date": close.last_valid_index().date().isoformat(),
        }
    prices = pd.concat(series, axis=1).sort_index()
    prices.index.name = "date"
    # A common trading date makes every stock and the benchmark synchronous.
    monthly = prices.dropna(how="any").resample("ME").last()
    expected_months = len(pd.period_range(start, (date.fromisoformat(end) - timedelta(days=1)).isoformat(), freq="M"))
    common_count = len(monthly.dropna())
    if common_count != expected_months or common_count < 61:
        raise DataError(f"Only {common_count} common month-end prices in {expected_months} requested months; need at least 61. Check listing dates and gaps; do not invent prices.")
    return prices, {
        "provider": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
        "method": "Ticker.history", "interval": "1d", "auto_adjust": True,
        "repair": False, "price_field": "Close (adjusted)", "symbols": details,
        "common_month_end_prices": common_count,
        "potential_common_monthly_returns": common_count - 1,
        "missing_daily_values_by_symbol": prices.isna().sum().to_dict(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", choices=["US", "HK"], required=True)
    parser.add_argument("--stocks", nargs=5, required=True, metavar="TICKER")
    parser.add_argument("--benchmark", required=True, metavar="TICKER")
    parser.add_argument("--start", default="2021-09-01", help="Inclusive first month, YYYY-MM-01")
    parser.add_argument("--end", default="2026-10-01", help="Exclusive first day after the last completed month")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing downloaded input files")
    parser.add_argument("--riskfree-csv", type=Path, help="Explicitly use this saved cash CSV instead of downloading cash quotes")
    parser.add_argument("--riskfree-date-kind", choices=["actual-observation", "source-period"],
                        help="Declare whether supplied dates are actual observations or source-period labels")
    parser.add_argument("--riskfree-currency", choices=["HKD", "USD"],
                        help="Explicitly confirm supplied cash CSV currency; must match --market")
    args = parser.parse_args(argv)
    args.stocks = [symbol.strip().upper() for symbol in args.stocks]
    args.benchmark = args.benchmark.strip().upper()
    if len(set(args.stocks)) != 5 or args.benchmark in args.stocks:
        parser.error("Choose five different stocks and a separate benchmark.")
    if not all(args.stocks + [args.benchmark]):
        parser.error("Ticker names cannot be blank.")
    if args.market == "HK" and not all(s.endswith(".HK") for s in args.stocks + [args.benchmark]):
        parser.error("Hong Kong Yahoo tickers must use the .HK suffix, for example 0700.HK.")
    if args.market == "US" and any(s.endswith(".HK") for s in args.stocks + [args.benchmark]):
        parser.error("A Hong Kong ticker cannot use the USD Treasury benchmark.")
    if args.riskfree_csv:
        if not args.riskfree_date_kind or not args.riskfree_currency:
            parser.error("--riskfree-csv requires both --riskfree-date-kind and --riskfree-currency.")
        if args.riskfree_currency != MARKETS[args.market]:
            parser.error("The declared supplied cash currency must match the selected market.")
    elif args.riskfree_date_kind or args.riskfree_currency:
        parser.error("Supplied cash declarations require --riskfree-csv.")
    try:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    except ValueError:
        parser.error("Use valid dates in YYYY-MM-DD format.")
    if start.day != 1 or end.day != 1 or start >= end:
        parser.error("Start and exclusive end must be first-of-month dates, with start before end.")
    if (end.year - start.year) * 12 + end.month - start.month < 61:
        parser.error("Request at least 61 price months to obtain 60 monthly returns.")
    if end > date.today().replace(day=1):
        parser.error("The end date would include an incomplete or future month.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_names = ["prices_daily.csv", "riskfree_daily.csv", "metadata.json"]
    existing = [name for name in output_names if (args.out_dir / name).exists()]
    if existing and not args.overwrite:
        print("Already present: " + ", ".join(existing) + ". Use a new --out-dir, or --overwrite to refresh deliberately.", file=sys.stderr)
        return 1
    try:
        # Validate an explicitly supplied file before making price-network requests.
        supplied = (load_supplied_rates(args.riskfree_csv, args.riskfree_currency,
                                        args.riskfree_date_kind, args.start, args.end)
                    if args.riskfree_csv else None)
        prices, price_info = fetch_prices(args.stocks + [args.benchmark], MARKETS[args.market], args.start, args.end,
                                          args.out_dir / ".yfinance_cache")
        if supplied:
            rates, rate_info, cached = supplied
        else:
            rates, rate_info, cached = (fetch_fred if args.market == "US" else fetch_hkma)(args.start, args.end)
        # Quotes available before each return month must exist; never backfill future rates.
        monthly_dates = prices.dropna(how="any").resample("ME").last().dropna().index[1:]
        for month_end in monthly_dates:
            month_start = month_end.to_period("M").start_time
            prior_month_start = (month_end.to_period("M") - 1).start_time
            if not ((rates["date"] >= prior_month_start) & (rates["date"] < month_start)).any():
                raise DataError(f"No bill quote in the immediately preceding calendar month for {month_end:%Y-%m}; do not carry a stale quote forward.")
        rate_info["first_observation_date"] = rates["date"].min().date().isoformat()
        rate_info["last_observation_date"] = rates["date"].max().date().isoformat()
        metadata = {
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "market": args.market, "currency": MARKETS[args.market],
            "stocks": args.stocks, "benchmark": args.benchmark,
            "start_inclusive": args.start, "end_exclusive": args.end,
            "prices": price_info, "riskfree": rate_info,
            "pandas_version": pd.__version__,
            "cache_policy": "Saved CSV inputs freeze this download. No automatic source substitution or stale-cache fallback. Source-response files preserve the bill download. --overwrite requests a deliberate refresh.",
            "monthly_bill_convention": "For month t, use the last quoted value in the immediately preceding calendar month; annual_yield_percent / 1200. Date kind is disclosed in riskfree.date_kind; no stale-rate carry-forward.",
            "missing_value_policy": "No price/rate is generated or filled. Missing observations remain missing; missing bill-source rows are omitted.",
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        source_cache = args.out_dir / "source_cache"
        source_cache.mkdir(exist_ok=True)
        for filename, raw in cached.items():
            (source_cache / filename).write_bytes(raw)
        # All downloads and validations succeed before replacing the three final files.
        prices.to_csv(args.out_dir / "prices_daily.csv", date_format="%Y-%m-%d")
        rates.to_csv(args.out_dir / "riskfree_daily.csv", index=False, date_format="%Y-%m-%d")
        metadata["sha256"] = {name: hashlib.sha256((args.out_dir / name).read_bytes()).hexdigest()
                              for name in output_names[:2]}
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Download stopped: {exc}\nNo substitute or synthetic data were produced. Existing input CSVs were not deliberately replaced before download validation.", file=sys.stderr)
        return 1
    print(f"Saved adjusted prices, annual-percent bill yields and metadata to {args.out_dir.resolve()}")
    print(f"Common month-end prices: {price_info['common_month_end_prices']}; potential monthly returns: {price_info['potential_common_monthly_returns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
