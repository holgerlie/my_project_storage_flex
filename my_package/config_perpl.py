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

    # Price process selector.
    # "lognormal_ou"  — single-factor lognormal OU (default, fastest)
    # "schwartz_smith" — two-factor Schwartz-Smith (seasonal spread model)
    # "heston"         — Heston stochastic vol (QuantLib QE scheme, slowest)
    process_model  = "lognormal_ou",
)

# ── Calendar / Dates ─────────────────────────────────────────────────────────
CALENDAR = dict(
    valuation_date = date(2026, 4, 17),   # today (pricing date)
    start_date     = date(2026, 4, 17),   # storage contract start
    end_date       = date(2027, 4, 17),   # storage contract end (1 gas year)
)

# ── Market Parameters ────────────────────────────────────────────────────────
# Shared market observables used across all models and the discounting layer.
# theta is the flat forward price fallback used by MarketParams.forward_price()
# when no forward_curve is provided; it doubles as the OU long-run mean target
# but is market data, not a process parameter.
# Process-specific dynamics (kappa, sigma) live in OU below.
MARKET = dict(
    spot_price     = 35.50,   # EUR/MWh, current TTF front month
    theta          = 38.00,   # long-run mean / flat forward fallback (EUR/MWh)
    risk_free_rate = 0.035,   # continuous discount rate (EUR, 3.5%)

    # Optional: forward curve override (date -> EUR/MWh).
    # If provided, forward_price(d) interpolates here; otherwise returns theta.
    # Set to None to use flat theta.
    forward_curve  = {
        date(2026,  5,  1): 35.80,
        date(2026,  6,  1): 33.50,
        date(2026,  7,  1): 32.00,
        date(2026,  8,  1): 31.50,
        date(2026,  9,  1): 33.00,
        date(2026, 10,  1): 37.50,
        date(2026, 11,  1): 41.00,
        date(2026, 12,  1): 44.00,
        date(2027,  1,  1): 45.50,
        date(2027,  2,  1): 44.00,
        date(2027,  3,  1): 40.00,
        date(2027,  4,  1): 36.00,
    },
)

# ── Lognormal OU Process Parameters ──────────────────────────────────────────
# Used when SIMULATION["process_model"] == "lognormal_ou".
# spot_price is shared with MARKET; the forward_price() callable is supplied
# by MarketParams and injected by the process factory — not duplicated here.
#
# Calibration guidance:
#   kappa : AR(1) on daily log-returns: kappa = -ln(phi)/dt
#           where phi is the lag-1 autocorrelation of log-returns
#   sigma : annualised std of log-return residuals from the same regression
OU = dict(
    kappa = 2.0,    # mean-reversion speed (yr-1); half-life = ln(2)/kappa ~ 4 months
    sigma = 0.45,   # annual log-price volatility
)

# ── Storage Facility ─────────────────────────────────────────────────────────
STORAGE = dict(
    # Working gas volumes (MWh)
    min_inventory     = 0,          # cushion gas not tradeable
    max_inventory     = 2_000_000,  # total working gas capacity

    # Starting inventory on valuation date
    initial_inventory = 800_000,    # MWh (partially filled, typical spring)

    # End-of-contract inventory constraint
    terminal_min_inventory = 0,     # must leave at least this in store
    terminal_max_inventory = 2_000_000,

    # Penalty multiplier for terminal inventory constraint violations (EUR/MWh shortfall
    # expressed as a multiple of spot price).  Applied in both the greedy MC pipeline
    # (StorageDispatcher / compute_path_npv) and the LSMC backward and forward passes.
    # Must be defined once here to guarantee consistency across all three valuations.
    # Economic interpretation: cost of sourcing / disposing of gas at short notice
    # relative to spot price (e.g. 5x = 5 times the prevailing spot rate).
    terminal_penalty_multiplier = 5.0,

    # Injection constraints (MWh/day); rates vary seasonally
    # Format: (month_start, month_end_inclusive) -> max_rate
    injection_rate_schedule = {
        (4, 9):  80_000,    # summer injection season
        (10, 3): 20_000,    # limited injection in winter
    },

    # Withdrawal constraints (MWh/day)
    withdrawal_rate_schedule = {
        (10, 3): 120_000,   # peak winter withdrawal
        (4, 9):  40_000,    # limited summer withdrawal
    },

    # Efficiency losses
    injection_efficiency   = 0.985,  # 1.5% compression / fuel loss on inject
    withdrawal_efficiency  = 0.998,  # 0.2% fuel use on withdraw

    # Variable costs (EUR/MWh injected or withdrawn)
    injection_cost_per_mwh   = 0.08,
    withdrawal_cost_per_mwh  = 0.05,

    # Fixed daily operating cost (EUR/day)
    daily_fixed_cost = 500.0,
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

    # Minimum spread (EUR/MWh) required to trigger inject/withdraw action.
    # Suppresses noise-driven churn on stochastic MC paths.
    dead_band = 0.20,

    # LP solver backend for LPIntrinsicOptimiser.
    # "highs" is the default scipy/HiGHS solver — fast, exact, open-source.
    # Other valid options: "highs-ds" (dual simplex), "highs-ipm" (interior point).
    lp_solver = "highs",
)

# ── Two-Factor Schwartz-Smith Parameters ────────────────────────────────
# Used when SIMULATION["process_model"] == "schwartz_smith".
# spot_price is shared with MARKET and injected by the process factory.
#
# Calibration guidance:
#   kappa_xi  : fit to ATM implied vol term structure (fast decay = high kappa)
#   sigma_xi  : short-dated implied vol level
#   sigma_eta : long-dated (1y+) implied vol level
#   rho       : correlation from historical regression of monthly vs annual returns
#   mu_eta    : risk-neutral drift = 0 (no risk premium in Q measure)
#   eta_0     : ln(F_LT) where F_LT is the last pillar of the forward curve
#               — set to None to derive automatically from MARKET["forward_curve"]
SCHWARTZ = dict(
    eta_0      = None,    # EUR/MWh; None → last pillar of MARKET["forward_curve"]
    kappa_xi   = 2.0,     # short-term mean-reversion speed (yr⁻¹), half-life ~4 months
    sigma_xi   = 0.35,    # short-term factor volatility
    mu_eta     = 0.0,     # long-term drift (risk-neutral Q measure: 0)
    sigma_eta  = 0.15,    # long-term factor volatility
    rho        = -0.20,   # factor correlation dW₁·dW₂
)

# ── Heston Stochastic Volatility Parameters ───────────────────────────────
# Used when SIMULATION["process_model"] == "heston".
# spot_price and risk_free_rate are shared with MARKET.
#
# Calibration guidance:
#   v0, theta_v : current and long-run variance = sigma² if using MARKET["sigma"]
#                 set to None to derive automatically from MARKET["sigma"]
#   kappa       : fit jointly with xi from implied vol surface curvature
#   xi          : vol-of-vol controls vol smile steepness
#   rho         : typically negative for energy (spot-vol leverage effect)
#   Feller:     2*kappa*theta_v > xi² for a.s. positive variance
HESTON = dict(
    v0         = None,    # initial variance; None → MARKET["sigma"]**2
    kappa      = 2.0,     # variance mean-reversion speed (yr⁻¹)
    theta_v    = None,    # long-run variance; None → MARKET["sigma"]**2
    xi         = 0.40,    # vol-of-vol
    rho        = -0.60,   # correlation spot-variance
    use_sobol  = False,   # Sobol quasi-random (slower setup, better convergence)
)

# ── LSMC Parameters ──────────────────────────────────────────────────────────
# Used when GasStorageSimulator runs the LSMC engine alongside the greedy MC.
# All three valuations (LP intrinsic, greedy MC, LSMC) run in the same call.
LSMC = dict(
    # Action discretisation: candidates are {k/n_actions * max_rate} for k=1..n
    # 5 levels = {20%, 40%, 60%, 80%, 100%} of max rate plus idle (0%).
    # Increase to 10 for higher accuracy; runtime scales linearly.
    n_actions   = 5,

    # Regression polynomial degree for price and inventory basis terms.
    # 3 = cubic (recommended for storage: captures asymmetric boundary effects).
    poly_degree = 3,

    # Polynomial family: "power" | "laguerre" | "chebyshev"
    # "power" is the most interpretable; "laguerre" is the original LS choice.
    basis_type  = "power",

    # Include inventory x price (and inventory x state2) cross terms.
    # Strongly recommended for storage: captures that high inventory is worth
    # more when prices are high (withdrawal option is in-the-money).
    cross_terms = True,

    # Paths used for backward regression fitting pass.
    # Remaining paths are used for forward pricing (out-of-sample).
    # Set equal to SIMULATION["n_paths"] to use all paths for both passes.
    # Rule of thumb: fit_paths >= 5,000 for stable regression at poly_degree=3.
    fit_paths   = 5_000,

    # Ridge regularisation coefficient (L2 penalty on regression coefficients).
    # Prevents ill-conditioning when many basis functions are used.
    # Typical range: 1e-6 to 1e-3. Set to 0.0 to disable.
    regularise  = 1e-4,
)

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT = dict(
    histogram_bins   = 80,
    percentiles      = [5, 10, 25, 50, 75, 90, 95],
    save_path_sample = True,     # save a sample of simulated price paths to CSV
    n_sample_paths   = 20,       # how many paths to export for inspection
    output_dir       = ".",      # relative to run_simulation.py location
)
