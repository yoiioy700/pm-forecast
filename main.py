"""Offline quantile forecaster for make.fun's polymarket-nvidia-ceo-forecast-01 bounty."""

import json
import math
import os
import statistics

Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]


def clamp(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def mean_abs_change(prices: list[float]) -> float:
    window = prices[-120:]
    if len(window) < 10:
        return 0.0
    return sum(abs(window[i] - window[i - 1]) for i in range(1, len(window))) / (len(window) - 1)


def forecast_case(case: dict) -> dict:
    horizon = int(case.get("horizon_minutes", 240))
    markets = case["markets"]
    stats = []
    for m in markets:
        prices = m["prices"]
        x = float(prices[-1])
        v = mean_abs_change(prices)
        stats.append((m["id"], x, v, len(prices)))

    ref = statistics.median(v for _, _, v, _ in stats) if stats else 0.0
    forecasts = []
    for market_id, x, v, n_prices in stats:
        if n_prices < 240 or ref <= 1e-9:
            scale = 1.0
        else:
            scale = clamp((max(v, 1e-5) / ref) ** 0.2, 0.6, 1.8)

        rows = []
        for h in range(1, horizon + 1):
            s = 0.0012 * scale * math.sqrt(h)
            row = []
            for z_tau in Z:
                q_tau = clamp(x + z_tau * s, 0.0, 1.0)
                if row:
                    q_tau = max(row[-1], q_tau)
                row.append(q_tau)
            rows.append(row)
        forecasts.append({"id": market_id, "q": rows})

    return {"forecasts": forecasts}


def main() -> None:
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith(".in"):
            continue
        case_id = fname[:-3]
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, f"{case_id}.out")
        with open(in_path, "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
