"""Offline quantile forecaster for the Polymarket NVIDIA CEO make.fun bounty."""

import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

QUANTILES_Z: tuple[float, ...] = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
BASE_SIGMA: float = 0.0012
SCALE_MIN: float = 0.6
SCALE_MAX: float = 1.8
SHORT_HISTORY: int = 240
VOL_WINDOW: int = 120
MIN_VOL_SAMPLES: int = 10
VOL_FLOOR: float = 1e-5
REF_EPS: float = 1e-9
SCALE_EXPONENT: float = 0.2


def clip(value: float, lower: float, upper: float) -> float:
    """Clamp a floating-point number into [lower, upper]."""
    return max(lower, min(upper, value))


def recent_volatility(prices: list[float]) -> float:
    """Compute mean absolute 1-minute price change over the recent window."""
    tail = prices[-VOL_WINDOW:]
    if len(tail) < MIN_VOL_SAMPLES:
        return 0.0
    diffs = (abs(tail[idx] - tail[idx - 1]) for idx in range(1, len(tail)))
    return sum(diffs) / (len(tail) - 1)


def forecast_market_rows(
    prices: list[float],
    ref_vol: float,
    horizon: int,
) -> list[list[float]]:
    """Return horizon rows of [q05, q25, q50, q75, q95] for a single market."""
    last_price = float(prices[-1])
    vol = recent_volatility(prices)

    if len(prices) < SHORT_HISTORY or ref_vol <= REF_EPS:
        scale = 1.0
    else:
        ratio = max(vol, VOL_FLOOR) / ref_vol
        scale = clip(ratio ** SCALE_EXPONENT, SCALE_MIN, SCALE_MAX)

    rows: list[list[float]] = []
    for step in range(1, horizon + 1):
        sigma = BASE_SIGMA * scale * math.sqrt(step)
        quantiles: list[float] = []
        for z_val in QUANTILES_Z:
            q_val = clip(last_price + z_val * sigma, 0.0, 1.0)
            if quantiles and q_val < quantiles[-1]:
                q_val = quantiles[-1]
            quantiles.append(q_val)
        rows.append(quantiles)
    return rows


def process_case(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Generate forecasts for all markets in a single evaluation case."""
    horizon = int(payload.get("horizon_minutes", 240))
    markets: list[dict[str, Any]] = payload["markets"]
    vols = [recent_volatility(m["prices"]) for m in markets]
    ref_vol = float(statistics.median(vols)) if vols else 0.0

    forecasts: list[dict[str, Any]] = [
        {
            "id": str(market["id"]),
            "q": forecast_market_rows(market["prices"], ref_vol, horizon),
        }
        for market in markets
    ]
    return {"forecasts": forecasts}


def main() -> None:
    """Iterate sorted .in files in MAKEFUN_INPUT_DIR and write .out files."""
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for in_file in sorted(input_dir.glob("*.in")):
        case_data: dict[str, Any] = json.loads(in_file.read_text(encoding="utf-8"))
        out_data = process_case(case_data)
        out_file = output_dir / f"{in_file.stem}.out"
        out_file.write_text(json.dumps(out_data), encoding="utf-8")


if __name__ == "__main__":
    main()
