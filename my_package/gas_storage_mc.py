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
    """Market and price-process parameters."""
    spot_price: float
    kappa: float
    theta: float
    sigma: float
    risk_free_rate: float
    forward_curve: Optional[Dict[date, float]] = None

    # Interpolated forward function (built lazily)
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


@dataclass
class SimulationParams:
    """Monte Carlo simulation settings."""
    n_paths: int = 10_000
    seed: Optional[int] = 42
    antithetic: bool = True
    n_workers: int = 1


@dataclass
class OptimiserParams:
    """Dispatch optimiser settings."""
    strategy: str = "rolling_intrinsic"   # "threshold" or "rolling_intrinsic"
    inject_threshold: float = 33.0
    withdraw_threshold: float = 42.0
    forward_lookback_days: int = 30


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
        storage = self.storage
        fwd_ref = self._fwd_ref[step]

        if strategy == "threshold":
            if spot < self.opt.inject_threshold:
                return "inject"
            elif spot > self.opt.withdraw_threshold:
                return "withdraw"
            return "idle"

        elif strategy == "rolling_intrinsic":
            spread = fwd_ref - spot
            # Inject if current price is cheaper than forward reference
            # Use a small dead-band to avoid churning
            dead_band = 0.20   # EUR/MWh
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
# Main Simulator
# ════════════════════════════════════════════════════════════════════════════

class GasStorageSimulator:
    """
    Full Monte Carlo gas storage valuation.

    Parameters
    ----------
    storage  : StorageParams
    market   : MarketParams
    sim      : SimulationParams
    opt      : OptimiserParams
    start    : date   — contract start
    end      : date   — contract end

    Usage
    -----
    >>> sim = GasStorageSimulator(storage, market, sim_params, opt_params, start, end)
    >>> results = sim.run()
    >>> print(results.summary())
    >>> results.plot_npv_histogram("storage_npv.png")
    """

    def __init__(
        self,
        storage: StorageParams,
        market: MarketParams,
        sim: SimulationParams,
        opt: OptimiserParams,
        start: date,
        end: date,
    ):
        self.storage = storage
        self.market = market
        self.sim = sim
        self.opt = opt
        self.date_grid = _build_date_grid(start, end)

    def run(self) -> "SimulationResults":
        """Execute Monte Carlo simulation and return results."""
        print(f"Simulating {self.sim.n_paths:,} paths over "
              f"{len(self.date_grid)} days...")

        # 1. Generate price paths
        process = LognormalOUProcess(self.market, self.sim)
        price_paths = process.simulate(self.date_grid)  # (n_paths, n_steps)

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

        print(f"Simulation complete. Mean NPV: EUR {npvs.mean():,.0f}")

        return SimulationResults(
            npvs=npvs,
            date_grid=self.date_grid,
            sample_prices=np.array(sample_prices),
            sample_inventories=np.array(sample_inventories),
            market=self.market,
            sim_params=self.sim,
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
        Bumps sigma by ±`bump` (e.g. 0.01 = 1 vol point).
        """
        base_sigma = self.market.sigma
        self.market.sigma = base_sigma + bump
        r_up = self.run()
        self.market.sigma = base_sigma - bump
        r_dn = self.run()
        self.market.sigma = base_sigma   # restore
        return (r_up.mean_npv - r_dn.mean_npv) / (2.0 * bump)

    def theta_sensitivity(self, bump: float = 1.0) -> float:
        """
        Mean-reversion level sensitivity: dNPV / d_theta.
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
    ):
        self.npvs = npvs
        self.date_grid = date_grid
        self.sample_prices = sample_prices
        self.sample_inventories = sample_inventories
        self.market = market
        self.sim_params = sim_params

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
            f"  OU kappa          : {self.market.kappa:>12.3f}",
            f"  OU theta          : EUR {self.market.theta:>10.2f} / MWh",
            f"  OU sigma          : {self.market.sigma:>12.1%}",
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
        ax_hist.set_title(
            f"Gas Storage NPV Distribution  |  {len(self.npvs):,} Monte Carlo Paths\n"
            f"Mean: EUR {self.mean_npv/1e6:.2f}M  |  "
            f"Std: EUR {self.std_npv/1e6:.2f}M  |  "
            f"P5–P95: EUR {pct_vals[5]/1e6:.2f}M – EUR {pct_vals[95]/1e6:.2f}M",
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
