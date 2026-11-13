"""Offline quantile forecaster for the Polymarket NVIDIA CEO make.fun bounty."""

import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

QUANTILES_Z: tuple[float, ...] = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
SIGMA_CENTRE: float = 0.40
SIGMA_TAIL: float = 0.60
HORIZON_EXP: float = 0.45
JUMP: float = 0.006
SHRINK: float = 600.0
VOL_WINDOW: int = 240
MIN_VOL_SAMPLES: int = 10
LOGIT_EPS: float = 1e-4


def clip(value: float, lower: float, upper: float) -> float:
    """Clamp a floating-point number into [lower, upper]."""
    return max(lower, min(upper, value))


def logit(price: float) -> float:
    """Map a price in (0, 1) to the real line."""
    p = clip(price, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return math.log(p / (1.0 - p))


def logistic(value: float) -> float:
    """Inverse of logit."""
    return 1.0 / (1.0 + math.exp(-value))


def recent_volatility(prices: list[float]) -> float:
    """Standard deviation of 1-minute logit changes over the recent window."""
    tail = [logit(p) for p in prices[-VOL_WINDOW:]]
    if len(tail) < MIN_VOL_SAMPLES:
        return 0.0
    diffs = [tail[idx] - tail[idx - 1] for idx in range(1, len(tail))]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return math.sqrt(max(var, 1e-12))


def forecast_market_rows(
    prices: list[float],
    ref_vol: float,
    horizon: int,
) -> list[list[float]]:
    """Return horizon rows of [q05, q25, q50, q75, q95] for a single market."""
    count = len(prices)
    sigma = recent_volatility(prices)
    sigma = math.sqrt((count * sigma * sigma + SHRINK * ref_vol * ref_vol) / (count + SHRINK))
    eta = logit(float(prices[-1]))

    rows: list[list[float]] = []
    for step in range(1, horizon + 1):
        base = sigma * (step ** HORIZON_EXP)
        quantiles: list[float] = []
        for idx, z_val in enumerate(QUANTILES_Z):
            k = SIGMA_CENTRE if abs(z_val) < 1.0 else SIGMA_TAIL
            q_val = logistic(eta + z_val * base * k)
            if idx in (0, 4):
                q_val += JUMP if z_val > 0 else -JUMP
            q_val = clip(q_val, 0.0, 1.0)
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
