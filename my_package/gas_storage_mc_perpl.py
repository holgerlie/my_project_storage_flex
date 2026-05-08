"""
gas_storage_mc.py — Gas Storage Monte Carlo Valuation Engine
=============================================================
Implements a Monte Carlo simulator for a natural gas storage facility using:

  • Lognormal Ornstein-Uhlenbeck (mean-reverting) price process
  • Seasonal injection / withdrawal rate constraints
  • Efficiency losses and variable costs
  • Rolling-intrinsic or threshold-based dispatch optimiser
  • NPV distribution with percentile statistics and sensitivities

Dependencies
------------
    pip install QuantLib numpy pandas scipy matplotlib tqdm

QuantLib is used for:
  - Date arithmetic and calendar conventions
  - Discount factors (flat yield term structure)
  - Random number generation (Sobol / pseudo-RNG with antithetic)

Author : Holger Lienemann (energy trading / quant analysis)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.interpolate import interp1d
from scipy.optimize import linprog
import scipy.sparse as sp

# LSMC engine — imported at module level; guarded so gas_storage_mc remains
# usable without lsmc.py during incremental development.
try:
    from lsmc import LSMCParams, LSMCOptimiser, LSMCResult
    _HAS_LSMC = True
except ImportError:
    _HAS_LSMC = False
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import QuantLib as ql
    _HAS_QL = True
except ImportError:
    _HAS_QL = False
    warnings.warn(
        "QuantLib not found — falling back to pure NumPy discount factors. "
        "Install via: pip install QuantLib",
        ImportWarning,
    )


# ════════════════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StorageParams:
    """Physical and contractual parameters of the storage facility."""
    min_inventory: float
    max_inventory: float
    initial_inventory: float
    terminal_min_inventory: float
    terminal_max_inventory: float
    injection_rate_schedule: Dict[Tuple[int, int], float]
    withdrawal_rate_schedule: Dict[Tuple[int, int], float]
    injection_efficiency: float
    withdrawal_efficiency: float
    injection_cost_per_mwh: float
    withdrawal_cost_per_mwh: float
    daily_fixed_cost: float

    def max_injection_rate(self, month: int) -> float:
        """Return max injection rate (MWh/day) for a given month."""
        for (m_start, m_end), rate in self.injection_rate_schedule.items():
            if _month_in_range(month, m_start, m_end):
                return rate
        return 0.0

    def max_withdrawal_rate(self, month: int) -> float:
        """Return max withdrawal rate (MWh/day) for a given month."""
        for (m_start, m_end), rate in self.withdrawal_rate_schedule.items():
            if _month_in_range(month, m_start, m_end):
                return rate
        return 0.0


@dataclass
class MarketParams:
    """
    Shared market observables — spot price, discount rate, forward curve, flat fallback.
    All values must be supplied explicitly — config.py is the single source of truth.

    kappa and sigma have been moved to OUParams (config.OU) because they are
    process parameters, not market observables.  theta is kept here because it
    serves as the flat forward price fallback in forward_price() and is market data.

    forward_curve : Optional[Dict[date, float]]
        Monthly TTF forward pillars {date: EUR/MWh}. Pass None to use flat theta.
        Kept Optional here because None is a legitimate runtime value, not a default.
    """
    spot_price    : float
    theta         : float
    risk_free_rate: float
    forward_curve : Optional[Dict[date, float]]

    # Internal lazy-initialisation fields — not user-facing parameters.
    # field(init=False) means they are excluded from __init__ and unpacking.
    _fwd_func: Optional[object] = field(default=None, init=False, repr=False)

    def forward_price(self, d: date) -> float:
        """
        Return the forward price for date d.
        Uses the forward_curve if available (linear interpolation),
        otherwise returns the long-run mean theta.
        """
        if self.forward_curve is None:
            return self.theta

        if self._fwd_func is None:
            dates_sorted = sorted(self.forward_curve.keys())
            t0 = dates_sorted[0]
            xs = [(d_ - t0).days for d_ in dates_sorted]
            ys = [self.forward_curve[d_] for d_ in dates_sorted]
            self._fwd_func = interp1d(
                xs, ys,
                kind="linear",
                fill_value=(ys[0], ys[-1]),
                bounds_error=False,
            )
            self._fwd_ref = t0

        t = (d - self._fwd_ref).days
        return float(self._fwd_func(t))

    def print_forward_curve(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Return the full daily forward curve as a DataFrame, using the same
        interpolation logic as forward_price(d).

        Parameters
        ----------
        start : date, optional
            First date of the daily grid. Defaults to the first pillar date
            in forward_curve (or today if no curve).
        end : date, optional
            Last date of the daily grid. Defaults to the last pillar date
            in forward_curve.

        Returns
        -------
        pd.DataFrame with columns:
            Date          — calendar date
            Forward_Price — interpolated EUR/MWh price for that date
            Is_Pillar     — True if the date is an input pillar point
        """
        # Determine date range
        if self.forward_curve is not None:
            pillars = sorted(self.forward_curve.keys())
            start = start or pillars[0]
            end   = end   or pillars[-1]
        else:
            # Flat curve — use provided range or a 1-year window from today
            from datetime import date as date_cls
            start = start or date_cls.today()
            end   = end   or (start + timedelta(days=365))
            pillars = []

        # Build daily grid and evaluate forward_price for each date
        grid = _build_date_grid(start, end)
        pillar_set = set(pillars)

        rows = [
            {
                "Date":          d,
                "Forward_Price": round(self.forward_price(d), 4),
                "Is_Pillar":     d in pillar_set,
            }
            for d in grid
        ]

        df = pd.DataFrame(rows).set_index("Date")

        # Pretty-print to console
        print(f"{'Date':<14} {'Forward (EUR/MWh)':>18} {'Pillar':>8}")
        print("-" * 44)
        for d, row in df.iterrows():
            marker = " <--" if row["Is_Pillar"] else ""
            print(f"{str(d):<14} {row['Forward_Price']:>18.4f}{marker}")

        return df


@dataclass
class SimulationParams:
    """
    Monte Carlo simulation settings.
    All values must be supplied explicitly — config.py is the single source of truth.

    process_model : str
        Selects the stochastic price process used for MC path generation.
        "lognormal_ou"   — single-factor lognormal OU (exact analytic, fast)
        "schwartz_smith" — two-factor Schwartz-Smith (exact analytic, seasonal)
        "heston"         — Heston stochastic vol (QuantLib QE scheme)
        Process-specific parameters are held in ProcessParams (see below).
    """
    n_paths       : int
    seed          : Optional[int]
    antithetic    : bool
    n_workers     : int
    process_model : str


@dataclass
class OptimiserParams:
    """
    Dispatch optimiser settings.
    All values must be supplied explicitly — config.py is the single source of truth.

    dead_band : float
        Minimum spread (EUR/MWh) required to trigger inject/withdraw in
        rolling_intrinsic mode. MC dispatcher uses 0.20 to suppress
        noise-driven churn on stochastic paths. IntrinsicValuator
        constructs a local instance with dead_band=0.0.
    """
    strategy             : str
    inject_threshold     : float
    withdraw_threshold   : float
    forward_lookback_days: int
    dead_band            : float
    lp_solver            : str    # HiGHS method string for LPIntrinsicOptimiser


# ════════════════════════════════════════════════════════════════════════════
# ProcessParams — process-specific parameters (single source: config.py)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OUParams:
    """
    Lognormal OU process parameters.
    Constructed from config.OU in run_simulation.build_params().

    Fields used by model
    --------------------
    lognormal_ou : kappa, sigma
    """
    kappa : float   # mean-reversion speed (yr-1)
    sigma : float   # annual log-price volatility


@dataclass
class ProcessParams:
    """
    Holds all process-specific parameters for the three supported models.
    Constructed in run_simulation.py via build_params() by merging OU,
    SCHWARTZ and HESTON dicts from config.py.

    Fields used per model
    ---------------------
    lognormal_ou   : kappa_ou, sigma_ou
    schwartz_smith : eta_0, kappa_xi, sigma_xi, mu_eta, sigma_eta, rho
    heston         : v0, kappa_heston, theta_v, xi, rho_heston, use_sobol

    Naming conventions
    ------------------
    OU fields use the _ou suffix; Heston fields use _heston, to make every
    field unambiguously model-scoped and to prevent accidental cross-use.
    The process factory maps these to the correct constructor argument names.
    """
    # Lognormal OU fields
    kappa_ou   : float
    sigma_ou   : float
    # Schwartz-Smith fields
    eta_0      : Optional[float]  # EUR/MWh; None → resolved from forward curve
    kappa_xi   : float
    sigma_xi   : float
    mu_eta     : float
    sigma_eta  : float
    rho        : float            # Schwartz-Smith factor correlation
    # Heston fields
    v0         : Optional[float]  # None → resolved from sigma_ou**2
    kappa_heston: float
    theta_v    : Optional[float]  # None → resolved from sigma_ou**2
    xi         : float
    rho_heston : float            # Heston spot-variance correlation
    use_sobol  : bool


# ════════════════════════════════════════════════════════════════════════════
# Process factory
# ════════════════════════════════════════════════════════════════════════════

def build_process(
    model      : str,
    market     : "MarketParams",
    sim        : SimulationParams,
    proc_params: ProcessParams,
    date_grid  : List[date],
):
    """
    Instantiate and return the correct process class for the given model string.

    Parameters
    ----------
    model       : sim.process_model — "lognormal_ou" | "schwartz_smith" | "heston"
    market      : MarketParams — provides spot_price, kappa, sigma, risk_free_rate,
                  forward_price() callable.  Shared params are read from here;
                  never duplicated in ProcessParams.
    sim         : SimulationParams — provides n_paths, seed, antithetic
    proc_params : ProcessParams — process-specific params from config.SCHWARTZ / HESTON
    date_grid   : List[date] — used only for eta_0 auto-resolution (Schwartz-Smith)

    Returns
    -------
    A process instance with a .simulate(date_grid) -> PathBundle method.

    Raises
    ------
    ValueError if model string is not recognised.
    """
    from processes import (
        LognormalOUProcess,
        TwoFactorSchwartzProcess,
        HestonProcess,
    )

    if model == "lognormal_ou":
        return LognormalOUProcess(
            spot_price = market.spot_price,
            kappa      = proc_params.kappa_ou,
            sigma      = proc_params.sigma_ou,
            fwd_func   = market.forward_price,
            n_paths    = sim.n_paths,
            seed       = sim.seed,
            antithetic = sim.antithetic,
        )

    elif model == "schwartz_smith":
        # eta_0: last pillar of the forward curve if not explicitly set
        if proc_params.eta_0 is not None:
            eta_0 = proc_params.eta_0
        elif market.forward_curve is not None:
            eta_0 = market.forward_curve[max(market.forward_curve)]
        else:
            eta_0 = market.theta   # fallback to flat long-run mean

        return TwoFactorSchwartzProcess(
            spot_price = market.spot_price,
            eta_0      = eta_0,
            kappa_xi   = proc_params.kappa_xi,
            sigma_xi   = proc_params.sigma_xi,
            mu_eta     = proc_params.mu_eta,
            sigma_eta  = proc_params.sigma_eta,
            rho        = proc_params.rho,
            n_paths    = sim.n_paths,
            seed       = sim.seed,
            antithetic = sim.antithetic,
        )

    elif model == "heston":
        sigma_sq = proc_params.sigma_ou ** 2   # OU sigma as default variance proxy
        return HestonProcess(
            spot_price     = market.spot_price,
            risk_free_rate = market.risk_free_rate,
            v0             = proc_params.v0      if proc_params.v0      is not None else sigma_sq,
            kappa          = proc_params.kappa_heston,
            theta_v        = proc_params.theta_v if proc_params.theta_v is not None else sigma_sq,
            xi             = proc_params.xi,
            rho            = proc_params.rho_heston,
            n_paths        = sim.n_paths,
            seed           = sim.seed,
            use_sobol      = proc_params.use_sobol,
        )

    else:
        raise ValueError(
            f"Unknown process_model {model!r}. "
            "Choose: 'lognormal_ou', 'schwartz_smith', 'heston'."
        )


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _month_in_range(month: int, start: int, end: int) -> bool:
    """Check whether `month` falls in the seasonal band [start, end], wrapping over year-end."""
    if start <= end:
        return start <= month <= end
    else:  # wraps: e.g. Oct(10) – Mar(3)
        return month >= start or month <= end


def _build_date_grid(start: date, end: date) -> List[date]:
    """Return a daily grid from start (inclusive) to end (inclusive)."""
    grid = []
    d = start
    while d <= end:
        grid.append(d)
        d += timedelta(days=1)
    return grid


def _discount_factor(r: float, t_years: float) -> float:
    """Continuous compounding discount factor."""
    return np.exp(-r * t_years)


# ════════════════════════════════════════════════════════════════════════════
# Price Process — Lognormal Ornstein-Uhlenbeck
# ════════════════════════════════════════════════════════════════════════════

class LognormalOUProcess:
    """
    Simulate log-price following an Ornstein-Uhlenbeck process:

        d(ln S) = kappa * (ln(theta_t) - ln(S)) * dt + sigma * dW

    where theta_t is the time-varying forward price (or constant theta).
    This ensures mean reversion around the forward curve without allowing
    negative prices.

    Exact discretisation (no Euler bias):
        ln S_{t+dt} = ln(theta_t) + (ln S_t - ln(theta_t)) * exp(-kappa*dt)
                      + sigma * sqrt((1 - exp(-2*kappa*dt)) / (2*kappa)) * Z
    """

    def __init__(self, market: MarketParams, sim: SimulationParams):
        self.market = market
        self.sim = sim

    def simulate(
        self,
        date_grid: List[date],
    ) -> np.ndarray:
        """
        Simulate price paths on the given date grid.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps)
            Simulated daily prices.
        """
        m = self.market
        s = self.sim
        n_steps = len(date_grid)

        # RNG setup
        rng = np.random.default_rng(s.seed)

        # Determine effective number of independent paths
        n_ind = s.n_paths // 2 if s.antithetic else s.n_paths

        # Draw standard normals: shape (n_ind, n_steps-1)
        Z = rng.standard_normal((n_ind, n_steps - 1))

        if s.antithetic:
            Z = np.vstack([Z, -Z])   # antithetic variates

        # Pre-compute forward prices and OU parameters for each step
        log_theta = np.array(
            [np.log(m.forward_price(d)) for d in date_grid]
        )

        # Build paths using exact OU discretisation
        paths = np.zeros((s.n_paths, n_steps))
        paths[:, 0] = np.log(m.spot_price)

        for i in range(n_steps - 1):
            dt = (date_grid[i + 1] - date_grid[i]).days / 365.25
            e = np.exp(-m.kappa * dt)
            std = m.sigma * np.sqrt((1.0 - np.exp(-2.0 * m.kappa * dt)) / (2.0 * m.kappa))

            paths[:, i + 1] = (
                log_theta[i + 1]
                + (paths[:, i] - log_theta[i + 1]) * e
                + std * Z[:, i]
            )

        return np.exp(paths)   # back to price space


# ════════════════════════════════════════════════════════════════════════════
# Dispatch Optimiser
# ════════════════════════════════════════════════════════════════════════════

class StorageDispatcher:
    """
    Greedy dispatch optimiser. For each simulated path and time step,
    decides inject / withdraw / idle to maximise path cash flows.

    Strategies
    ----------
    "threshold"
        Inject if spot < inject_threshold, withdraw if spot > withdraw_threshold.

    "rolling_intrinsic"
        Compare spot to the average forward price over the next
        `forward_lookback_days` days. Inject if spot < fwd average (cheap),
        withdraw if spot > fwd average (expensive).
    """

    def __init__(
        self,
        storage: StorageParams,
        market: MarketParams,
        opt: OptimiserParams,
        date_grid: List[date],
    ):
        self.storage = storage
        self.market = market
        self.opt = opt
        self.date_grid = date_grid

        # Pre-build forward reference for rolling intrinsic
        self._fwd_ref = self._build_fwd_reference()

    def _build_fwd_reference(self) -> np.ndarray:
        """
        For each date in the grid, compute the average forward price
        over the next `forward_lookback_days` calendar days.
        """
        lb = self.opt.forward_lookback_days
        refs = np.zeros(len(self.date_grid))
        for i, d in enumerate(self.date_grid):
            fwd_prices = [
                self.market.forward_price(d + timedelta(days=j))
                for j in range(1, lb + 1)
            ]
            refs[i] = np.mean(fwd_prices)
        return refs

    def dispatch_path(
        self,
        prices: np.ndarray,        # shape (n_steps,)
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run dispatch for a single price path.

        Returns
        -------
        inventory  : np.ndarray (n_steps,)  — daily inventory level
        net_volume : np.ndarray (n_steps,)  — positive=inject, negative=withdraw
        terminal_penalty : float            — large penalty if terminal constraint violated
        """
        n = len(prices)
        inventory = np.zeros(n)
        inventory[0] = self.storage.initial_inventory
        net_volume = np.zeros(n)

        for i in range(n - 1):
            d = self.date_grid[i]
            month = d.month
            inv = inventory[i]
            spot = prices[i]

            max_inj = self.storage.max_injection_rate(month)
            max_wit = self.storage.max_withdrawal_rate(month)

            # Space available for injection / withdrawal
            room_to_fill = self.storage.max_inventory - inv
            room_to_empty = inv - self.storage.min_inventory

            inj_cap = min(max_inj, room_to_fill)
            wit_cap = min(max_wit, room_to_empty)

            # Dispatch decision
            action = self._decide(spot, i, inv)

            if action == "inject" and inj_cap > 0:
                vol = inj_cap
                net_volume[i] = vol
                inventory[i + 1] = inv + vol * self.storage.injection_efficiency
            elif action == "withdraw" and wit_cap > 0:
                vol = -wit_cap
                net_volume[i] = vol
                inventory[i + 1] = inv + vol   # withdrawal: remove vol (already negative)
            else:
                net_volume[i] = 0.0
                inventory[i + 1] = inv

            # Clamp to bounds (floating point safety)
            inventory[i + 1] = np.clip(
                inventory[i + 1],
                self.storage.min_inventory,
                self.storage.max_inventory,
            )

        # Terminal inventory — sell remaining gas at final price
        terminal_inventory = inventory[-1]
        terminal_penalty = 0.0
        if terminal_inventory < self.storage.terminal_min_inventory:
            terminal_penalty = (
                (self.storage.terminal_min_inventory - terminal_inventory)
                * prices[-1] * 5.0   # penalty = 5x market price per MWh shortfall
            )
        # Any gas above terminal_max is penalised too (can't store it)
        if terminal_inventory > self.storage.terminal_max_inventory:
            terminal_penalty += (
                (terminal_inventory - self.storage.terminal_max_inventory)
                * prices[-1] * 5.0
            )

        return inventory, net_volume, terminal_penalty

    def _decide(self, spot: float, step: int, inventory: float) -> str:
        """Return 'inject', 'withdraw', or 'idle'."""
        strategy = self.opt.strategy
        fwd_ref  = self._fwd_ref[step]

        if strategy == "threshold":
            if spot < self.opt.inject_threshold:
                return "inject"
            elif spot > self.opt.withdraw_threshold:
                return "withdraw"
            return "idle"

        elif strategy == "rolling_intrinsic":
            spread = fwd_ref - spot
            # CHANGE 2: dead_band now read from OptimiserParams instead of
            # being hardcoded. MC paths use 0.20 EUR/MWh; intrinsic uses 0.0.
            dead_band = self.opt.dead_band
            if spread > dead_band:
                return "inject"
            elif spread < -dead_band:
                return "withdraw"
            return "idle"

        return "idle"


# ════════════════════════════════════════════════════════════════════════════
# Cash Flow Calculator
# ════════════════════════════════════════════════════════════════════════════

def compute_path_npv(
    prices: np.ndarray,
    net_volumes: np.ndarray,
    inventory: np.ndarray,
    terminal_penalty: float,
    date_grid: List[date],
    storage: StorageParams,
    market: MarketParams,
) -> float:
    """
    Compute the NPV of a single path.

    Cash flows:
      • Injection: pay spot price × injected volume + injection cost
      • Withdrawal: receive spot price × withdrawn volume (after efficiency) − withdrawal cost
      • Daily fixed cost (negative)
      • Terminal: sell remaining inventory at final spot price
      • Terminal penalty if constraint violated
    """
    t0 = date_grid[0]
    npv = 0.0

    for i, d in enumerate(date_grid[:-1]):
        vol = net_volumes[i]
        spot = prices[i]
        t = (d - t0).days / 365.25
        df = _discount_factor(market.risk_free_rate, t)

        if vol > 0:   # injection: we buy gas
            cf = -(spot * vol + storage.injection_cost_per_mwh * vol)
        elif vol < 0:   # withdrawal: we sell gas
            sold_vol = abs(vol) * storage.withdrawal_efficiency
            cf = spot * sold_vol - storage.withdrawal_cost_per_mwh * abs(vol)
        else:
            cf = 0.0

        cf -= storage.daily_fixed_cost
        npv += df * cf

    # Terminal: liquidate remaining inventory at final day price
    t_terminal = (date_grid[-1] - t0).days / 365.25
    df_terminal = _discount_factor(market.risk_free_rate, t_terminal)
    terminal_inventory = inventory[-1]
    npv += df_terminal * (terminal_inventory * prices[-1] - terminal_penalty)

    return npv


# ════════════════════════════════════════════════════════════════════════════
# Intrinsic Valuation
# ════════════════════════════════════════════════════════════════════════════

class IntrinsicResult:
    """
    Container for intrinsic valuation output.

    Attributes
    ----------
    npv       : float         — intrinsic NPV (EUR)
    breakdown : pd.DataFrame  — daily dispatch schedule with cash flows
    date_grid : list of date
    """

    def __init__(self, npv: float, breakdown: pd.DataFrame, date_grid: List[date]):
        self.npv       = npv
        self.breakdown = breakdown
        self.date_grid = date_grid

    def summary(self) -> str:
        """Return a formatted summary of the intrinsic dispatch."""
        inj  = self.breakdown[self.breakdown["Action"] == "Inject"]
        wit  = self.breakdown[self.breakdown["Action"] == "Withdraw"]
        idle = self.breakdown[self.breakdown["Action"] == "Idle"]
        lines = [
            "=" * 55,
            "  Intrinsic Valuation",
            "=" * 55,
            f"  NPV                 : EUR {self.npv:>14,.0f}",
            "-" * 55,
            f"  Injection days      : {len(inj):>14,}",
            f"  Withdrawal days     : {len(wit):>14,}",
            f"  Idle days           : {len(idle):>14,}",
            f"  Total injected MWh  : {inj['Volume_MWh'].sum():>14,.0f}",
            f"  Total withdrawn MWh : {abs(wit['Volume_MWh'].sum()):>14,.0f}",
            f"  Total fixed costs   : EUR {self.breakdown['Fixed_CF_EUR'].sum():>10,.0f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def save_csv(self, path: str = "intrinsic_dispatch.csv") -> str:
        """Save the daily dispatch schedule to CSV."""
        self.breakdown.to_csv(path)
        print(f"Intrinsic dispatch schedule saved → {path}")
        return path


class LPIntrinsicOptimiser:
    """
    Globally optimal intrinsic storage valuation via Linear Programming.

    Replaces the greedy rolling-intrinsic dispatcher in IntrinsicValuator
    with a full LP over the known forward curve, solved exactly using
    scipy.optimize.linprog with the HiGHS backend.

    Problem formulation
    -------------------
    Decision variables (per day t = 0 .. T-1):
        u[t]  > 0  injection volume (MWh)
        v[t]  > 0  withdrawal volume (MWh)

    Objective (maximise discounted NPV):
        sum_t  disc[t] * [ F[t]*v[t]*eta_wit - F[t]*u[t]
                           - c_inj*u[t] - c_wit*v[t] - c_fix ]

    Constraints:
        Inventory evolution  (equality at every step)
        Inventory bounds     I_min <= I[t] <= I_max
        Rate limits          0 <= u[t] <= inj_cap[t],  0 <= v[t] <= wit_cap[t]
        Terminal bounds      I_min_terminal <= I[T] <= I_max_terminal

    All constraints are linear in (u, v) so the problem is a pure LP.
    HiGHS solves a 366-step daily problem in < 10 ms.

    Sparse matrices are used throughout so the implementation scales
    to multi-year contracts and intra-day granularity without modification.

    Parameters
    ----------
    storage   : StorageParams
    market    : MarketParams
    date_grid : list of date
    lp_solver : str   HiGHS method string passed to linprog (default 'highs')
    """

    def __init__(
        self,
        storage   : StorageParams,
        market    : MarketParams,
        date_grid : List[date],
        lp_solver : str = "highs",
    ):
        self.storage   = storage
        self.market    = market
        self.date_grid = date_grid
        self.lp_solver = lp_solver
        self.T         = len(date_grid) - 1   # number of decision steps

    # ── Public interface ─────────────────────────────────────────────────────

    def solve(
        self,
        fwd_prices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the LP and return the optimal dispatch.

        Parameters
        ----------
        fwd_prices : np.ndarray (T+1,)
            Deterministic forward price at each date in date_grid.

        Returns
        -------
        inventory        : np.ndarray (T+1,)  end-of-day inventory levels
        net_volume       : np.ndarray (T,)    positive=inject, negative=withdraw
        terminal_penalty : float              always 0.0 (terminal handled as
                                              hard LP constraint)
        """
        T   = self.T
        st  = self.storage
        mkt = self.market

        # ── 1. Build discount factors and rate-limit arrays ───────────────
        t0   = self.date_grid[0]
        disc = np.array([
            _discount_factor(mkt.risk_free_rate,
                             (self.date_grid[t] - t0).days / 365.25)
            for t in range(T)
        ])

        inj_cap = np.array([
            st.max_injection_rate(self.date_grid[t].month) for t in range(T)
        ])
        wit_cap = np.array([
            st.max_withdrawal_rate(self.date_grid[t].month) for t in range(T)
        ])

        F = fwd_prices[:T]   # forward prices at decision steps

        # ── 2. Objective vector c (negated: linprog minimises) ────────────
        # Injection: cost = F[t] + c_inj per MWh paid
        # Withdrawal: revenue = F[t]*eta_wit - c_wit per MWh received
        # Fixed cost: constant per day (does not affect optimum)
        c_inj_vec = disc * (F + st.injection_cost_per_mwh)
        c_wit_vec = disc * (-(F * st.withdrawal_efficiency - st.withdrawal_cost_per_mwh))

        # x = [u_0..u_{T-1}, v_0..v_{T-1}]  shape (2T,)
        c_obj = np.concatenate([c_inj_vec, c_wit_vec])

        # ── 3. Variable bounds (rate limits, non-negativity) ──────────────
        bounds = (
            [(0.0, float(cap)) for cap in inj_cap]
            + [(0.0, float(cap)) for cap in wit_cap]
        )

        # ── 4. Sparse inequality constraints for inventory bounds ─────────
        #
        # Inventory at step t+1:
        #   I[t+1] = I[0] + eta_inj * sum_{s<=t} u[s] - sum_{s<=t} v[s]
        #
        # I_min <= I[t+1] <= I_max  for t = 0..T-1
        # Terminal: I_min_term <= I[T] <= I_max_term  (included in t=T-1)
        #
        # Expressed as two sets of inequalities:
        #   -eta_inj * L @ u  +  L @ v  <=  I[0] - I_min[t]   (lower bound)
        #    eta_inj * L @ u  -  L @ v  <=  I_max[t] - I[0]   (upper bound)
        # where L is lower-triangular ones matrix.

        L = sp.tril(np.ones((T, T)), format="csc")   # lower triangular (T x T)

        # I_min and I_max per step — use terminal bounds on the last step
        I_min_vec = np.full(T, float(st.min_inventory))
        I_max_vec = np.full(T, float(st.max_inventory))
        I_min_vec[T - 1] = float(st.terminal_min_inventory)
        I_max_vec[T - 1] = float(st.terminal_max_inventory)

        I0 = float(st.initial_inventory)

        # Lower bound: I[t+1] >= I_min  =>  -eta*L*u + L*v <= I0 - I_min
        A_lb_u = -st.injection_efficiency * L
        A_lb_v =  L
        b_lb   = I0 - I_min_vec

        # Upper bound: I[t+1] <= I_max  =>   eta*L*u - L*v <= I_max - I0
        A_ub_u =  st.injection_efficiency * L
        A_ub_v = -L
        b_ub   = I_max_vec - I0

        # Stack both sets: shape (2T, 2T)
        A_ub_mat = sp.bmat([
            [A_lb_u, A_lb_v],
            [A_ub_u, A_ub_v],
        ], format="csc")
        b_ub_vec = np.concatenate([b_lb, b_ub])

        # ── 5. Solve ──────────────────────────────────────────────────────
        result = linprog(
            c       = c_obj,
            A_ub    = A_ub_mat,
            b_ub    = b_ub_vec,
            bounds  = bounds,
            method  = self.lp_solver,
            options = {"disp": False, "presolve": True},
        )

        if result.status != 0:
            raise RuntimeError(
                f"LP solver did not find optimal solution: "
                f"status={result.status} — {result.message}"
            )

        # ── 6. Reconstruct inventory and net_volume arrays ────────────────
        u_opt = result.x[:T]   # optimal injection schedule
        v_opt = result.x[T:]   # optimal withdrawal schedule

        inventory  = np.zeros(T + 1)
        net_volume = np.zeros(T)
        inventory[0] = I0

        for t in range(T):
            inj = u_opt[t]
            wit = v_opt[t]
            net_volume[t]      = inj - wit
            inventory[t + 1]   = (
                inventory[t]
                + inj * st.injection_efficiency
                - wit
            )

        # Terminal penalty is always 0.0 — the LP enforces terminal bounds
        # as hard constraints so violations are structurally impossible.
        terminal_penalty = 0.0

        return inventory, net_volume, terminal_penalty

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def print_lp_summary(self, u_opt: np.ndarray, v_opt: np.ndarray) -> None:
        """Print a brief LP solution summary."""
        print(f"  LP inject days      : {(u_opt > 1e-3).sum()}")
        print(f"  LP withdraw days    : {(v_opt > 1e-3).sum()}")
        print(f"  LP total injected   : {u_opt.sum():,.0f} MWh")
        print(f"  LP total withdrawn  : {v_opt.sum():,.0f} MWh")


class IntrinsicValuator:
    """
    Computes the intrinsic value of a gas storage facility.

    The intrinsic value is the NPV achievable by optimally dispatching
    against the known forward curve today with no uncertainty. It is
    computed as a single deterministic forward-pass — no simulation required.

    This is the lower bound of total storage value. The difference between
    the Monte Carlo mean NPV and the intrinsic value is the extrinsic
    (time / optionality) value.

    Parameters
    ----------
    storage   : StorageParams
    market    : MarketParams  — forward_curve and risk_free_rate used
    date_grid : list of date
    """

    # CHANGE 4: _LOOKBACK kept here as the single source of truth for the
    # lookback window — passed into OptimiserParams inside run() rather than
    # used by a local _build_fwd_reference(). Eliminates the duplicate.
    _LOOKBACK = 30

    def __init__(
        self,
        storage:   StorageParams,
        market:    MarketParams,
        date_grid: List[date],
        lp_solver: str = "highs",
    ):
        self.storage   = storage
        self.market    = market
        self.date_grid = date_grid
        self.lp_solver = lp_solver

    def run(self) -> IntrinsicResult:
        """
        Execute the intrinsic valuation using the LP global optimiser.

        CHANGE 5: StorageDispatcher (greedy rolling-intrinsic) replaced by
        LPIntrinsicOptimiser. The LP solves globally over the full forward
        curve in a single pass, guaranteeing the true intrinsic upper bound.
        This eliminates the negative extrinsic value produced by the greedy
        dispatcher, which myopically sold the initial inventory at low summer
        prices instead of holding for winter.

        Steps
        -----
        1. Build the daily forward price array from market.forward_price(d)
        2. Construct LPIntrinsicOptimiser with lp_solver from OptimiserParams
        3. Call lp_opt.solve(fwd_prices) — global LP optimisation
        4. Compute discounted NPV via compute_path_npv
        5. Build daily breakdown DataFrame

        Returns
        -------
        IntrinsicResult
        """
        # Step 1 — deterministic forward price path
        fwd_prices = np.array(
            [self.market.forward_price(d) for d in self.date_grid]
        )

        # Step 2 — LP global optimiser (replaces greedy StorageDispatcher)
        lp_opt = LPIntrinsicOptimiser(
            self.storage, self.market, self.date_grid,
            lp_solver=self.lp_solver,
        )

        # Step 3 — global LP solve
        inventory, net_volume, terminal_penalty = lp_opt.solve(fwd_prices)

        # Step 4 — NPV
        npv = compute_path_npv(
            fwd_prices, net_volume, inventory,
            terminal_penalty, self.date_grid,
            self.storage, self.market,
        )

        # Step 5 — build daily breakdown
        breakdown = self._build_breakdown(fwd_prices, net_volume, inventory)

        return IntrinsicResult(npv=npv, breakdown=breakdown, date_grid=self.date_grid)

    def _build_breakdown(
        self,
        prices:     np.ndarray,
        net_volume: np.ndarray,
        inventory:  np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a daily breakdown DataFrame showing the full dispatch schedule.

        Columns
        -------
        Date           — calendar date
        Forward        — EUR/MWh forward price used for dispatch
        Action         — Inject / Withdraw / Idle
        Volume_MWh     — volume injected (+) or withdrawn (-) in MWh
        Inventory      — end-of-day inventory level (MWh)
        Trading_CF_EUR — undiscounted trading cash flow (EUR), excluding fixed costs
        Fixed_CF_EUR   — daily fixed cost (EUR), always -daily_fixed_cost
        Daily_CF_EUR   — total undiscounted daily cash flow = Trading_CF_EUR + Fixed_CF_EUR
        """
        rows = []
        for i, d in enumerate(self.date_grid[:-1]):
            vol  = net_volume[i]
            spot = prices[i]

            if vol > 0:
                action     = "Inject"
                trading_cf = -(spot * vol + self.storage.injection_cost_per_mwh * vol)
            elif vol < 0:
                action     = "Withdraw"
                sold_vol   = abs(vol) * self.storage.withdrawal_efficiency
                trading_cf = spot * sold_vol - self.storage.withdrawal_cost_per_mwh * abs(vol)
            else:
                action     = "Idle"
                trading_cf = 0.0

            fixed_cf = -self.storage.daily_fixed_cost

            rows.append({
                "Date":           d,
                "Forward":        round(spot, 4),
                "Action":         action,
                "Volume_MWh":     round(vol, 1),
                "Inventory":      round(inventory[i + 1], 1),
                "Trading_CF_EUR": round(trading_cf, 2),
                "Fixed_CF_EUR":   round(fixed_cf, 2),
                "Daily_CF_EUR":   round(trading_cf + fixed_cf, 2),
            })

        return pd.DataFrame(rows).set_index("Date")


# ════════════════════════════════════════════════════════════════════════════
# Main Simulator
# ════════════════════════════════════════════════════════════════════════════

class GasStorageSimulator:
    """
    Full Monte Carlo gas storage valuation.

    Parameters
    ----------
    storage     : StorageParams
    market      : MarketParams
    sim         : SimulationParams    — includes process_model selector
    opt         : OptimiserParams
    proc_params : ProcessParams       — process-specific params (Schwartz / Heston)
    start       : date                — contract start
    end         : date                — contract end

    Usage
    -----
    >>> sim = GasStorageSimulator(storage, market, sim_params, opt_params, proc_params, start, end)
    >>> results = sim.run()
    >>> print(results.summary())
    >>> results.plot_npv_histogram("storage_npv.png")
    """

    def __init__(
        self,
        storage     : StorageParams,
        market      : MarketParams,
        sim         : SimulationParams,
        opt         : OptimiserParams,
        proc_params : ProcessParams,
        start       : date,
        end         : date,
        lsmc_params : Optional["LSMCParams"] = None,
    ):
        self.storage     = storage
        self.market      = market
        self.sim         = sim
        self.opt         = opt
        self.proc_params = proc_params
        self.lsmc_params = lsmc_params
        self.date_grid   = _build_date_grid(start, end)

    def run(self) -> "SimulationResults":
        """Execute Monte Carlo simulation and return results."""

        # ── 0. Intrinsic valuation first (deterministic, fast) ─────────────
        print("Computing intrinsic value...")
        intrinsic = IntrinsicValuator(
            self.storage, self.market, self.date_grid,
            lp_solver=self.opt.lp_solver,
        ).run()
        print(f"Intrinsic NPV  : EUR {intrinsic.npv:,.0f}")

        # ── 1. Monte Carlo ──────────────────────────────────────────────
        print(f"Simulating {self.sim.n_paths:,} paths over "
              f"{len(self.date_grid)} days...")

        # 1. Generate price paths via process factory
        # The factory reads sim.process_model and constructs the correct
        # process class, pulling shared params from market and sim and
        # process-specific params from proc_params.  price_paths is always
        # the .spots array of the returned PathBundle.
        process    = build_process(
            self.sim.process_model, self.market, self.sim,
            self.proc_params, self.date_grid,
        )
        bundle     = process.simulate(self.date_grid)
        price_paths = bundle.spots   # (n_paths, n_steps)

        # 2. Dispatch + NPV for each path
        dispatcher = StorageDispatcher(
            self.storage, self.market, self.opt, self.date_grid
        )

        npvs = np.zeros(self.sim.n_paths)
        sample_inventories = []
        sample_prices = []

        for p in range(self.sim.n_paths):
            inv, net_vol, terminal_penalty = dispatcher.dispatch_path(price_paths[p])
            npvs[p] = compute_path_npv(
                price_paths[p], net_vol, inv,
                terminal_penalty, self.date_grid,
                self.storage, self.market,
            )
            if p < 20:   # keep a sample for export
                sample_inventories.append(inv)
                sample_prices.append(price_paths[p])

        print(f"Simulation complete. Mean NPV  : EUR {npvs.mean():,.0f}")
        print(f"Extrinsic value                : EUR {npvs.mean() - intrinsic.npv:,.0f}")

        # ── 3. LSMC (optional) ──────────────────────────────────────────────────────────────
        lsmc_result = None
        if self.lsmc_params is not None:
            if not _HAS_LSMC:
                warnings.warn(
                    "lsmc.py not found — skipping LSMC. "
                    "Ensure lsmc.py is in the same directory.",
                    RuntimeWarning,
                )
            else:
                print("Running LSMC backward induction...")
                lsmc_engine = LSMCOptimiser(
                    self.storage, self.market, self.date_grid, self.lsmc_params
                )
                policy      = lsmc_engine.fit(bundle)
                lsmc_result = lsmc_engine.price(bundle, policy)
                print(f"LSMC NPV (mean) : EUR {lsmc_result.mean_npv:,.0f}")
                print(f"  {lsmc_result.r2_summary()}")

        return SimulationResults(
            npvs=npvs,
            date_grid=self.date_grid,
            sample_prices=np.array(sample_prices),
            sample_inventories=np.array(sample_inventories),
            market=self.market,
            sim_params=self.sim,
            proc_params=self.proc_params,
            intrinsic_result=intrinsic,
            lsmc_result=lsmc_result,
        )

    # ── Sensitivity / Greek-style analysis ──────────────────────────────────

    def delta(self, bump: float = 0.50) -> float:
        """
        Price delta: dNPV / dS0 estimated by bump-and-revalue.
        Bumps spot by ±`bump` EUR/MWh.
        """
        base_spot = self.market.spot_price
        self.market.spot_price = base_spot + bump
        r_up = self.run()
        self.market.spot_price = base_spot - bump
        r_dn = self.run()
        self.market.spot_price = base_spot   # restore
        return (r_up.mean_npv - r_dn.mean_npv) / (2.0 * bump)

    def vega(self, bump: float = 0.01) -> float:
        """
        Volatility vega: dNPV / d_sigma estimated by bump-and-revalue.
        Bumps sigma_ou by +/-`bump` (e.g. 0.01 = 1 vol point).
        For Schwartz-Smith, bump sigma_xi (the dominant short-term vol).
        For Heston, bump xi (vol-of-vol) and v0/theta_v simultaneously.
        Current implementation bumps sigma_ou which feeds the OU branch
        and the Heston v0/theta_v defaults.
        """
        base = self.proc_params.sigma_ou
        self.proc_params.sigma_ou = base + bump
        r_up = self.run()
        self.proc_params.sigma_ou = base - bump
        r_dn = self.run()
        self.proc_params.sigma_ou = base   # restore
        return (r_up.mean_npv - r_dn.mean_npv) / (2.0 * bump)

    def theta_sensitivity(self, bump: float = 1.0) -> float:
        """
        Long-run mean / flat forward level sensitivity: dNPV / d_theta.
        Bumps MarketParams.theta which is the flat forward fallback and the
        OU mean-reversion target when no forward_curve is provided.
        """
        base_theta = self.market.theta
        self.market.theta = base_theta + bump
        r_up = self.run()
        self.market.theta = base_theta - bump
        r_dn = self.run()
        self.market.theta = base_theta
        return (r_up.mean_npv - r_dn.mean_npv) / (2.0 * bump)


# ════════════════════════════════════════════════════════════════════════════
# Results Container
# ════════════════════════════════════════════════════════════════════════════

class SimulationResults:
    """Holds all simulation outputs and provides summary / plotting utilities."""

    def __init__(
        self,
        npvs: np.ndarray,
        date_grid: List[date],
        sample_prices: np.ndarray,
        sample_inventories: np.ndarray,
        market: MarketParams,
        sim_params: SimulationParams,
        proc_params: "ProcessParams",
        intrinsic_result: Optional["IntrinsicResult"] = None,
        lsmc_result: Optional["LSMCResult"] = None,
    ):
        self.npvs = npvs
        self.date_grid = date_grid
        self.sample_prices = sample_prices
        self.sample_inventories = sample_inventories
        self.market = market
        self.sim_params = sim_params
        self.proc_params = proc_params
        self.intrinsic_result = intrinsic_result
        self.lsmc_result      = lsmc_result

    @property
    def mean_npv(self) -> float:
        return float(np.mean(self.npvs))

    @property
    def std_npv(self) -> float:
        return float(np.std(self.npvs))

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.npvs, p))

    def summary(self, percentiles: List[int] = None) -> str:
        """Return a formatted summary string."""
        pcts = percentiles or [5, 10, 25, 50, 75, 90, 95]
        lines = [
            "=" * 55,
            "  Gas Storage Monte Carlo Valuation — Results",
            "=" * 55,
            f"  Paths simulated   : {len(self.npvs):>12,}",
            f"  Simulation period : {self.date_grid[0]} → {self.date_grid[-1]}",
            f"  Spot price (S0)   : EUR {self.market.spot_price:>10.2f} / MWh",
            f"  Process model     : {self.sim_params.process_model!r}",
            f"  OU kappa          : {self.proc_params.kappa_ou:>12.3f}",
            f"  OU sigma          : {self.proc_params.sigma_ou:>12.1%}",
            f"  Theta (fwd flat)  : EUR {self.market.theta:>10.2f} / MWh",
            "-" * 55,
            f"  Mean NPV          : EUR {self.mean_npv:>14,.0f}",
            f"  Std Dev           : EUR {self.std_npv:>14,.0f}",
            f"  Coeff of Variation: {self.std_npv / abs(self.mean_npv) if self.mean_npv else float('nan'):>12.1%}",
            "-" * 55,
            "  NPV Percentiles (EUR):",
        ]
        for p in pcts:
            lines.append(f"    P{p:<3d}            : EUR {self.percentile(p):>14,.0f}")
        lines.append("=" * 55)

        # ── Value decomposition ─────────────────────────────────────────────────
        if self.intrinsic_result is not None or self.lsmc_result is not None:
            lines += ["", "  Value Decomposition:", "-" * 55]

        if self.intrinsic_result is not None:
            iv  = self.intrinsic_result.npv
            ev  = self.mean_npv - iv
            pct = (ev / iv * 100) if iv != 0 else float("nan")
            lines += [
                f"  LP Intrinsic        : EUR {iv:>14,.0f}",
                f"  Greedy MC mean      : EUR {self.mean_npv:>14,.0f}",
                f"  Extrinsic (MC)      : EUR {ev:>14,.0f}  ({pct:.1f}%)",
            ]

        if self.lsmc_result is not None:
            lv = self.lsmc_result.mean_npv
            lines.append(f"  LSMC mean           : EUR {lv:>14,.0f}")
            if self.intrinsic_result is not None:
                ev_l  = lv - self.intrinsic_result.npv
                pct_l = (ev_l / iv * 100) if iv != 0 else float("nan")
                lines.append(
                    f"  Extrinsic (LSMC)    : EUR {ev_l:>14,.0f}  ({pct_l:.1f}%)"
                )
            lines.append(f"  {self.lsmc_result.r2_summary()}")

        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return NPV distribution as a one-column DataFrame."""
        return pd.DataFrame({"NPV_EUR": self.npvs})

    def plot_npv_histogram(
        self,
        save_path: str = "storage_npv.png",
        bins: int = 80,
        percentiles: List[int] = None,
    ) -> str:
        """
        Plot NPV distribution histogram with percentile markers.

        Returns the path to the saved file.
        """
        pcts = percentiles or [5, 25, 50, 75, 95]
        pct_vals = {p: self.percentile(p) for p in pcts}
        pct_colors = {5: "#e74c3c", 25: "#f39c12", 50: "#2ecc71", 75: "#3498db", 95: "#9b59b6"}

        fig, axes = plt.subplots(
            2, 1,
            figsize=(12, 9),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.set_facecolor("#1a1d27")

        ax_hist, ax_paths = axes

        # ── Histogram ───────────────────────────────────────────────────────
        n, bin_edges, patches = ax_hist.hist(
            self.npvs / 1e6,
            bins=bins,
            color="#4a90d9",
            edgecolor="#0f1117",
            linewidth=0.3,
            alpha=0.85,
        )

        # Colour bars by percentile zone
        for patch, left in zip(patches, bin_edges[:-1]):
            v = left * 1e6
            if v < pct_vals.get(5, -np.inf):
                patch.set_facecolor("#c0392b")
            elif v > pct_vals.get(95, np.inf):
                patch.set_facecolor("#8e44ad")

        # Percentile vertical lines — draw first, then add a legend box
        legend_handles = []
        for p, v in pct_vals.items():
            color = pct_colors.get(p, "white")
            line = ax_hist.axvline(
                v / 1e6, color=color, linewidth=1.6, linestyle="--", alpha=0.9,
                label=f"P{p}: EUR {v/1e6:.1f}M",
            )
            legend_handles.append(line)

        # ── BB5: Intrinsic line + extrinsic shading ──
        if self.intrinsic_result is not None:
            iv = self.intrinsic_result.npv
            ev = self.mean_npv - iv
            iv_line = ax_hist.axvline(
                iv / 1e6, color="#ffffff", linewidth=2.0,
                linestyle="-", alpha=0.9, zorder=5,
                label=f"Intrinsic: EUR {iv/1e6:.1f}M",
            )
            legend_handles.append(iv_line)
            # Shade the extrinsic band between intrinsic and MC mean
            ax_hist.axvspan(
                iv / 1e6, self.mean_npv / 1e6,
                alpha=0.12, color="#2ecc71", zorder=1,
            )
            # Label the extrinsic band
            mid_x = (iv + self.mean_npv) / 2 / 1e6
            ax_hist.text(
                mid_x, ax_hist.get_ylim()[1] * 0.55,
                f"Extrinsic\nEUR {ev/1e6:.1f}M",
                color="#2ecc71", fontsize=7.5, ha="center", va="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f1117",
                          edgecolor="#2ecc71", alpha=0.75),
            )

        ax_hist.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=8.5,
            labelcolor="white",
            facecolor="#0f1117",
            edgecolor="#444455",
            framealpha=0.85,
            handlelength=1.5,
        )

        ax_hist.set_xlabel("NPV (EUR million)", color="#cccccc", fontsize=11)
        ax_hist.set_ylabel("Frequency", color="#cccccc", fontsize=11)

        # Title — include value decomposition when intrinsic is available
        if self.intrinsic_result is not None:
            iv  = self.intrinsic_result.npv
            ev  = self.mean_npv - iv
            title_line2 = (
                f"Mean: EUR {self.mean_npv/1e6:.2f}M  |  "
                f"Intrinsic: EUR {iv/1e6:.2f}M  |  "
                f"Extrinsic: EUR {ev/1e6:.2f}M"
            )
        else:
            title_line2 = (
                f"Mean: EUR {self.mean_npv/1e6:.2f}M  |  "
                f"Std: EUR {self.std_npv/1e6:.2f}M  |  "
                f"P5–P95: EUR {pct_vals[5]/1e6:.2f}M – EUR {pct_vals[95]/1e6:.2f}M"
            )
        ax_hist.set_title(
            f"Gas Storage NPV Distribution  |  {len(self.npvs):,} Monte Carlo Paths\n"
            + title_line2,
            color="#e0e0e0", fontsize=12, pad=12,
        )
        ax_hist.tick_params(colors="#aaaaaa")
        ax_hist.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        ax_hist.spines[:].set_color("#333344")
        ax_hist.grid(axis="x", color="#333344", linewidth=0.5, linestyle=":")

        # ── Sample Price Paths ──────────────────────────────────────────────
        dates_dt = [pd.Timestamp(d) for d in self.date_grid]
        n_show = min(self.sample_prices.shape[0], 12)
        path_colors = ["#5dade2", "#85c1e9", "#7fb3d3", "#a9cce3"]
        for i in range(n_show):
            if i == 0:
                ax_paths.plot(dates_dt, self.sample_prices[i], color="#f1c40f", lw=1.8, alpha=0.95, zorder=3)
            else:
                c = path_colors[i % len(path_colors)]
                ax_paths.plot(dates_dt, self.sample_prices[i], color=c, lw=0.9, alpha=0.55, zorder=2)

        # Forward curve overlay
        if self.market.forward_curve:
            fwd_dates = sorted(self.market.forward_curve.keys())
            fwd_prices = [self.market.forward_curve[d] for d in fwd_dates]
            ax_paths.plot(
                [pd.Timestamp(d) for d in fwd_dates], fwd_prices,
                color="#e74c3c", lw=2.0, linestyle="-", alpha=0.9, label="Forward curve",
            )
            ax_paths.legend(fontsize=8, labelcolor="#cccccc", facecolor="#1a1d27", edgecolor="#333344")

        ax_paths.set_ylabel("Price (EUR/MWh)", color="#cccccc", fontsize=9)
        ax_paths.set_title("Sample Simulated Price Paths", color="#cccccc", fontsize=9, pad=6)
        ax_paths.tick_params(colors="#aaaaaa", labelsize=8)
        ax_paths.spines[:].set_color("#333344")
        ax_paths.grid(color="#333344", linewidth=0.4, linestyle=":")
        ax_paths.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%b %y")
        )
        plt.setp(ax_paths.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig.tight_layout(pad=1.8)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"NPV histogram saved → {save_path}")
        return save_path

    def save_sample_paths_csv(self, save_path: str = "sample_paths.csv") -> str:
        """Export sample simulated price paths to CSV for inspection."""
        df = pd.DataFrame(
            self.sample_prices.T,
            index=[str(d) for d in self.date_grid],
            columns=[f"Path_{i+1}" for i in range(self.sample_prices.shape[0])],
        )
        df.index.name = "Date"
        df.to_csv(save_path)
        print(f"Sample paths saved → {save_path}")
        return save_path

    def save_inventory_csv(self, save_path: str = "sample_inventory.csv") -> str:
        """Export sample inventory profiles to CSV."""
        df = pd.DataFrame(
            self.sample_inventories.T,
            index=[str(d) for d in self.date_grid],
            columns=[f"Path_{i+1}" for i in range(self.sample_inventories.shape[0])],
        )
        df.index.name = "Date"
        df.to_csv(save_path)
        print(f"Sample inventory saved → {save_path}")
        return save_path
