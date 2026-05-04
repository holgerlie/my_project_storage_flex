"""
config.py — Gas Storage Monte Carlo Simulator Configuration
============================================================
All parameters in one place. Calibrated to a typical TTF-linked
seasonal storage facility (e.g. German/Dutch underground storage).

Units:
  - Prices        : EUR/MWh
  - Volumes       : MWh
  - Rates         : MWh/day
  - Time          : calendar days
"""

from datetime import date

# ── Simulation ──────────────────────────────────────────────────────────────
SIMULATION = dict(
    n_paths        = 10_000,   # number of Monte Carlo paths
    seed           = 42,       # RNG seed (None = non-deterministic)
    antithetic     = True,     # use antithetic variates for variance reduction
    n_workers      = 4,        # parallel workers (set 1 to disable)
)

# ── Calendar / Dates ─────────────────────────────────────────────────────────
CALENDAR = dict(
    valuation_date = date(2026, 4, 17),   # today (pricing date)
    start_date     = date(2026, 4, 17),   # storage contract start
    end_date       = date(2027, 4, 17),   # storage contract end (1 gas year)
)

# ── Market Parameters ────────────────────────────────────────────────────────
# Ornstein-Uhlenbeck (mean-reverting) process:
#   dS = kappa*(theta - S)*dt + sigma*S*dW   [lognormal OU / Black-like]
#
# For a pure additive OU:
#   dX = kappa*(theta - X)*dt + sigma*dW
# We implement the lognormal variant (log-prices follow OU) to avoid negatives.
MARKET = dict(
    spot_price     = 35.00,   # EUR/MWh, current TTF front month
    kappa          = 10.0,     # mean-reversion speed (per year); ~6-month half-life
    theta          = 38.00,   # long-run mean price (EUR/MWh)
    sigma          = 0.001,    # annual volatility (log-price diffusion)
    risk_free_rate = 0.000,   # continuous discount rate (EUR, 3.5%)

    # Optional: forward curve override (date -> EUR/MWh).
    # If provided, the OU process is centred on the forward rather than theta.
    # Set to None to use flat theta.
    forward_curve  = {
        date(2026,  5,  1): 35.00,
        date(2026,  6,  1): 35.00,
        date(2026,  7,  1): 35.00,
        date(2026,  8,  1): 35.00,
        date(2026,  9,  1): 35.00,
        date(2026, 10,  1): 40.00,
        date(2026, 11,  1): 40.00,
        date(2026, 12,  1): 40.00,
        date(2027,  1,  1): 40.00,
        date(2027,  2,  1): 40.00,
        date(2027,  3,  1): 40.00,
        date(2027,  4,  1): 40.00,
    },
)

# ── Storage Facility ─────────────────────────────────────────────────────────
STORAGE = dict(
    # Working gas volumes (MWh)
    min_inventory     = 0,          # cushion gas not tradeable
    max_inventory     = 2_000_000,  # total working gas capacity

    # Starting inventory on valuation date
    initial_inventory = 0,    # MWh (partially filled, typical spring)

    # End-of-contract inventory constraint
    terminal_min_inventory = 0,     # must leave at least this in store
    terminal_max_inventory = 0,

    # Injection constraints (MWh/day); rates vary seasonally
    # Format: (month_start, month_end_inclusive) -> max_rate
    injection_rate_schedule = {
        (4, 9):  20_000,    # summer injection season
        (10, 3): 20_000,    # limited injection in winter
    },

    # Withdrawal constraints (MWh/day)
    withdrawal_rate_schedule = {
        (10, 3): 20_000,   # peak winter withdrawal
        (4, 9):  20_000,    # limited summer withdrawal
    },

    # Efficiency losses
    injection_efficiency   = 1.000,  # 1.5% compression / fuel loss on inject
    withdrawal_efficiency  = 1.000,  # 0.2% fuel use on withdraw

    # Variable costs (EUR/MWh injected or withdrawn)
    injection_cost_per_mwh   = 1.00,
    withdrawal_cost_per_mwh  = 1.00,

    # Fixed daily operating cost (EUR/day)
    daily_fixed_cost = 0.0,
)

# ── Optimiser ────────────────────────────────────────────────────────────────
OPTIMISER = dict(
    # Strategy: "threshold" (simple price band) or "rolling_intrinsic" (forward-based)
    strategy        = "rolling_intrinsic",

    # Threshold strategy: inject if price < inject_threshold, withdraw if > withdraw_threshold
    inject_threshold    = 33.0,   # EUR/MWh
    withdraw_threshold  = 42.0,   # EUR/MWh

    # Rolling intrinsic: compare spot vs. average forward over remaining horizon
    # Uses the forward_curve from MARKET; falls back to theta if not set.
    forward_lookback_days = 30,   # how many days ahead to average for forward reference
)

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT = dict(
    histogram_bins   = 80,
    percentiles      = [5, 10, 25, 50, 75, 90, 95],
    save_path_sample = True,     # save a sample of simulated price paths to CSV
    n_sample_paths   = 20,       # how many paths to export for inspection
    output_dir       = ".",      # relative to run_simulation.py location
)
