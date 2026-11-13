# polymarket-nvidia-ceo-forecast-01
Offline Python 3.13 stdlib-only quantile forecaster for make.fun's NVIDIA CEO Polymarket bounty.
Centers each market's 240-minute forecast at its last observed Yes price x.
Estimates 1-minute mean absolute change v over the last 120 prices (0 if <10) and takes the cross-market median ref.
Computes volatility scale = clamp((max(v, 1e-5) / ref) ** 0.2, 0.6, 1.8), or 1.0 when ref <= 1e-9 or history < 240 prices.
Projects [q05, q25, q50, q75, q95] as clamp(x + z * 0.0012 * scale * sqrt(h), 0, 1) with non-decreasing quantile rows.
