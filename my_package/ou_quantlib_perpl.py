"""
ou_quantlib.py — Lognormal OU Price Simulation using QuantLib
==============================================================
Standalone implementation of the mean-reverting lognormal Ornstein-Uhlenbeck
process using QuantLib's built-in process and path generation classes.

This module is intentionally independent of the Monte Carlo storage simulator
(gas_storage_mc.py). It can be used as a drop-in replacement for the price
simulation step, or studied in isolation to understand how QuantLib's
stochastic process framework works.

Process
-------
We model log-prices as an additive OU process:

    d(ln S) = kappa * (ln(theta_t) - ln(S)) * dt + sigma * dW

where theta_t is the time-varying forward price (seasonal TTF curve).

QuantLib implements this via ExtendedOrnsteinUhlenbeckProcess, which accepts
a Python callable for the time-varying mean-reversion level.

Two path generators are provided:
  - Pseudo-random  : GaussianPathGenerator  (Mersenne Twister)
  - Quasi-random   : GaussianSobolPathGenerator  (Sobol low-discrepancy)

The Sobol generator produces lower-discrepancy paths and converges faster
than pseudo-random for smooth payoff functions — useful when path counts
are limited.

Dependencies
------------
    pip install QuantLib numpy pandas scipy matplotlib

Usage
-----
    from ou_quantlib import LognormalOUQuantLib
    from datetime import date

    forward_curve = {
        date(2026,  5,  1): 35.80,
        date(2026, 10,  1): 37.50,
        date(2027,  1,  1): 45.50,
        date(2027,  4,  1): 36.00,
    }

    sim = LognormalOUQuantLib(
        spot_price    = 35.50,
        kappa         = 2.0,
        sigma         = 0.45,
        forward_curve = forward_curve,
        start         = date(2026, 4, 17),
        end           = date(2027, 4, 17),
        n_paths       = 10_000,
        seed          = 42,
        use_sobol     = False,
    )

    paths = sim.simulate()          # np.ndarray (n_paths, n_steps)
    sim.print_statistics(paths)
    sim.plot(paths, save_path="ou_ql_paths.png")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from typing import Dict, Optional
from scipy.interpolate import interp1d

import QuantLib as ql


# ════════════════════════════════════════════════════════════════════════════
# Helper — date conversion
# ════════════════════════════════════════════════════════════════════════════

def _to_ql_date(d: date) -> ql.Date:
    """Convert Python date to QuantLib Date."""
    return ql.Date(d.day, d.month, d.year)


def _build_date_grid(start: date, end: date) -> list[date]:
    """Daily grid from start to end inclusive."""
    grid, d = [], start
    while d <= end:
        grid.append(d)
        d += timedelta(days=1)
    return grid


# ════════════════════════════════════════════════════════════════════════════
# Forward curve interpolator
# ════════════════════════════════════════════════════════════════════════════

class ForwardCurve:
    """
    Linear interpolation of monthly TTF forward pillars onto a daily grid.

    The time axis is measured in years from the valuation date — this is the
    unit QuantLib's stochastic process uses internally.

    Parameters
    ----------
    pillars  : dict mapping date → EUR/MWh forward price
    ref_date : valuation / anchor date (t=0 in QuantLib time)
    theta    : fallback flat level used when pillars is None
    """

    def __init__(
        self,
        pillars: Optional[Dict[date, float]],
        ref_date: date,
        theta: float,
    ):
        self.pillars  = pillars
        self.ref_date = ref_date
        self.theta    = theta
        self._func: Optional[interp1d] = None

        if pillars is not None:
            dates_sorted = sorted(pillars.keys())
            # x-axis: years from ref_date (matching QuantLib's time convention)
            xs = [(d - ref_date).days / 365.25 for d in dates_sorted]
            ys = [np.log(pillars[d]) for d in dates_sorted]  # log-prices
            self._func = interp1d(
                xs, ys,
                kind="linear",
                fill_value=(ys[0], ys[-1]),
                bounds_error=False,
            )

    def log_forward(self, t: float) -> float:
        """
        Return ln(F(t)) for QuantLib time t (years from ref_date).
        This is the mean-reversion target for the log-price OU process.
        """
        if self._func is None:
            return np.log(self.theta)
        return float(self._func(t))

    def forward(self, t: float) -> float:
        """Return F(t) in EUR/MWh."""
        return np.exp(self.log_forward(t))


# ════════════════════════════════════════════════════════════════════════════
# Main class
# ════════════════════════════════════════════════════════════════════════════

class LognormalOUQuantLib:
    """
    Simulate TTF gas prices via QuantLib's ExtendedOrnsteinUhlenbeckProcess.

    The process evolves log-prices as an additive OU around the log-forward
    curve. The output is exponentiated back to price space, giving a
    lognormal OU (mean-reversion in log-space, positive prices guaranteed).

    Parameters
    ----------
    spot_price    : float        — current TTF spot (EUR/MWh)
    kappa         : float        — mean-reversion speed (per year)
    sigma         : float        — annual log-price volatility
    forward_curve : dict or None — {date: EUR/MWh} monthly pillars
    theta         : float        — fallback flat mean if no forward_curve
    start         : date         — simulation start date
    end           : date         — simulation end date
    n_paths       : int          — number of Monte Carlo paths
    seed          : int or None  — RNG seed (Mersenne Twister)
    use_sobol     : bool         — use Sobol quasi-random generator
    """

    def __init__(
        self,
        spot_price:    float,
        kappa:         float,
        sigma:         float,
        start:         date,
        end:           date,
        forward_curve: Optional[Dict[date, float]] = None,
        theta:         float = 35.0,
        n_paths:       int   = 10_000,
        seed:          Optional[int] = 42,
        use_sobol:     bool  = False,
    ):
        self.spot_price    = spot_price
        self.kappa         = kappa
        self.sigma         = sigma
        self.start         = start
        self.end           = end
        self.forward_curve = forward_curve
        self.theta         = theta
        self.n_paths       = n_paths
        self.seed          = seed
        self.use_sobol     = use_sobol

        # Daily grid and QuantLib time axis
        self.date_grid = _build_date_grid(start, end)
        self.n_steps   = len(self.date_grid) - 1   # number of intervals

        # Total time in years
        self.T = (end - start).days / 365.25

        # Forward curve interpolator
        self.fwd_curve = ForwardCurve(forward_curve, start, theta)

        # Set QuantLib valuation date
        ql.Settings.instance().evaluationDate = _to_ql_date(start)

    # ── QuantLib process construction ────────────────────────────────────────

    def _build_process(self) -> ql.ExtendedOrnsteinUhlenbeckProcess:
        """
        Build a QuantLib ExtendedOrnsteinUhlenbeckProcess.

        ExtendedOrnsteinUhlenbeckProcess signature:
            (speed, sigma, x0, function, intEps=1e-4)

        where:
          speed    = kappa  (mean-reversion speed, per year)
          sigma    = sigma  (volatility of the log-price diffusion)
          x0       = ln(S0) (initial log-price)
          function = callable t → mean-reversion level at time t
                     here this is ln(F(t)), the log-forward curve

        The process evolves:
            dX = speed * (function(t) - X) * dt + sigma * dW

        where X = ln(S). Exponentiating X gives S, which is always positive.
        """
        x0 = np.log(self.spot_price)

        # The mean-reversion target: a Python callable t → ln(F(t))
        # QuantLib calls this at each time step during path generation
        mean_reversion_target = self.fwd_curve.log_forward

        process = ql.ExtendedOrnsteinUhlenbeckProcess(
            self.kappa,             # speed
            self.sigma,             # sigma
            x0,                     # x0 = ln(S0)
            mean_reversion_target,  # time-varying level function
        )
        return process

    # ── Path generator construction ─────────────────────────────────────────

    def _build_generator(
        self,
        process: ql.ExtendedOrnsteinUhlenbeckProcess,
        time_grid: ql.TimeGrid,
    ):
        """
        Build a path generator backed by either:
          - GaussianPathGenerator      (Mersenne Twister pseudo-random)
          - GaussianSobolPathGenerator (Sobol quasi-random, lower discrepancy)

        Both generators accept a brownianBridge argument. We set it to False
        here — Brownian bridge conditioning is useful when the terminal value
        matters most (barrier options), but for storage dispatch we care about
        the full path shape, not just the endpoint.
        """
        if self.use_sobol:
            # Sobol low-discrepancy sequence
            # dimensionality = n_steps (one Sobol dimension per time step)
            uniform_rsg = ql.UniformLowDiscrepancySequenceGenerator(
                self.n_steps
            )
            gaussian_rsg = ql.GaussianLowDiscrepancySequenceGenerator(
                uniform_rsg
            )
            generator = ql.GaussianSobolPathGenerator(
                process,
                time_grid,
                gaussian_rsg,
                False,   # brownianBridge
            )
        else:
            # Mersenne Twister pseudo-random
            # seed=0 in QuantLib means non-deterministic; we pass seed+1
            # to avoid ql treating 0 as "no seed"
            ql_seed = (self.seed + 1) if self.seed is not None else 42
            uniform_rsg = ql.UniformRandomSequenceGenerator(
                self.n_steps,
                ql.UniformRandomGenerator(ql_seed),
            )
            gaussian_rsg = ql.GaussianRandomSequenceGenerator(uniform_rsg)
            generator = ql.GaussianPathGenerator(
                process,
                time_grid,
                gaussian_rsg,
                False,   # brownianBridge
            )
        return generator

    # ── Main simulation method ───────────────────────────────────────────────

    def simulate(self) -> np.ndarray:
        """
        Generate price paths using QuantLib's path generator.

        Internally:
          1. Build ExtendedOrnsteinUhlenbeckProcess in log-space
          2. Build TimeGrid (uniform daily steps)
          3. Build path generator (pseudo or quasi-random)
          4. Draw n_paths paths, extract log-price arrays
          5. Exponentiate to recover EUR/MWh price paths

        Returns
        -------
        np.ndarray of shape (n_paths, n_steps+1)
            Simulated daily TTF prices in EUR/MWh.
            Column 0 is always spot_price (t=0).
        """
        process   = self._build_process()
        time_grid = ql.TimeGrid(self.T, self.n_steps)
        generator = self._build_generator(process, time_grid)

        # Pre-allocate output: log-price paths in QL convention
        # time_grid has n_steps+1 nodes (including t=0)
        log_paths = np.empty((self.n_paths, self.n_steps + 1))

        for i in range(self.n_paths):
            sample = generator.next()   # SamplePath object
            path   = sample.value()     # QuantLib Path object
            # Extract all time nodes: path[0] = x0, path[1..n_steps] = evolved
            log_paths[i] = np.array([path[j] for j in range(self.n_steps + 1)])

        # Exponentiate: log-price → EUR/MWh price
        price_paths = np.exp(log_paths)

        rng_label = "Sobol" if self.use_sobol else "Mersenne Twister"
        print(
            f"QuantLib OU simulation complete — "
            f"{self.n_paths:,} paths × {self.n_steps + 1} steps "
            f"[{rng_label}]"
        )
        return price_paths

    # ── Statistics ───────────────────────────────────────────────────────────

    def print_statistics(
        self,
        paths: np.ndarray,
        percentiles: list[int] = None,
    ) -> pd.DataFrame:
        """
        Print and return summary statistics of the terminal price distribution
        (i.e. the simulated price on the last day of the contract).

        Parameters
        ----------
        paths       : output of simulate(), shape (n_paths, n_steps+1)
        percentiles : list of integer percentiles to report
        """
        pcts = percentiles or [5, 10, 25, 50, 75, 90, 95]
        terminal = paths[:, -1]

        rows = []
        print()
        print("=" * 50)
        print("  Terminal Price Distribution (EUR/MWh)")
        print(f"  Date: {self.end}")
        print("=" * 50)
        print(f"  Mean   : {terminal.mean():.4f}")
        print(f"  Std    : {terminal.std():.4f}")
        print(f"  Min    : {terminal.min():.4f}")
        print(f"  Max    : {terminal.max():.4f}")
        print("-" * 50)
        for p in pcts:
            v = float(np.percentile(terminal, p))
            print(f"  P{p:<3d}  : {v:.4f}")
            rows.append({"Percentile": p, "Price_EUR_MWh": v})
        print("=" * 50)

        # Also print forward curve target for reference
        t_end = (self.end - self.start).days / 365.25
        fwd_terminal = self.fwd_curve.forward(t_end)
        print(f"  Forward curve at end: {fwd_terminal:.4f}  (mean-reversion target)")
        print()

        return pd.DataFrame(rows).set_index("Percentile")

    # ── Plot ─────────────────────────────────────────────────────────────────

    def plot(
        self,
        paths: np.ndarray,
        save_path: str = "ou_ql_paths.png",
        n_show: int = 30,
    ) -> str:
        """
        Plot sample simulated price paths with forward curve overlay.

        Parameters
        ----------
        paths     : output of simulate()
        save_path : file path to save the chart
        n_show    : number of sample paths to display
        """
        dates_dt = [pd.Timestamp(d) for d in self.date_grid]

        fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.set_facecolor("#1a1d27")

        ax_paths, ax_dist = axes

        # ── Price paths ──────────────────────────────────────────────────────
        n_plot = min(n_show, paths.shape[0])
        path_colors = ["#5dade2", "#7fb3d3", "#85c1e9", "#a9cce3"]
        for i in range(n_plot):
            if i == 0:
                ax_paths.plot(dates_dt, paths[i], color="#f1c40f",
                              lw=1.8, alpha=0.95, zorder=3, label="Sample path (highlighted)")
            else:
                c = path_colors[i % len(path_colors)]
                ax_paths.plot(dates_dt, paths[i], color=c,
                              lw=0.8, alpha=0.45, zorder=2)

        # Forward curve overlay
        fwd_prices = [self.fwd_curve.forward((d - self.start).days / 365.25)
                      for d in self.date_grid]
        ax_paths.plot(dates_dt, fwd_prices, color="#e74c3c", lw=2.2,
                      linestyle="-", zorder=4, label="Forward curve (mean-reversion target)")

        # Spot price marker
        ax_paths.axhline(self.spot_price, color="#95a5a6", lw=1.0,
                         linestyle=":", alpha=0.7, label=f"Spot S₀ = {self.spot_price}")

        ax_paths.set_ylabel("Price (EUR/MWh)", color="#cccccc", fontsize=10)
        ax_paths.set_title(
            f"Lognormal OU Process — QuantLib ExtendedOrnsteinUhlenbeckProcess\n"
            f"κ={self.kappa}  σ={self.sigma}  S₀={self.spot_price}  "
            f"{'Sobol' if self.use_sobol else 'Mersenne Twister'}  "
            f"{self.n_paths:,} paths",
            color="#e0e0e0", fontsize=11, pad=10,
        )
        ax_paths.tick_params(colors="#aaaaaa")
        ax_paths.spines[:].set_color("#333344")
        ax_paths.grid(color="#333344", linewidth=0.4, linestyle=":")
        ax_paths.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        plt.setp(ax_paths.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_paths.legend(fontsize=8, labelcolor="#cccccc",
                        facecolor="#0f1117", edgecolor="#444455",
                        framealpha=0.85)

        # ── Terminal price distribution ───────────────────────────────────────
        terminal = paths[:, -1]
        ax_dist.hist(terminal, bins=60, color="#4a90d9",
                     edgecolor="#0f1117", linewidth=0.3, alpha=0.85)
        ax_dist.axvline(np.median(terminal), color="#2ecc71", lw=1.5,
                        linestyle="--", label=f"P50 = {np.median(terminal):.2f}")
        ax_dist.axvline(np.mean(terminal), color="#f39c12", lw=1.5,
                        linestyle="--", label=f"Mean = {np.mean(terminal):.2f}")
        ax_dist.axvline(fwd_prices[-1], color="#e74c3c", lw=1.5,
                        linestyle="-", label=f"Fwd = {fwd_prices[-1]:.2f}")
        ax_dist.set_xlabel(f"Terminal Price on {self.end} (EUR/MWh)",
                           color="#cccccc", fontsize=9)
        ax_dist.set_ylabel("Frequency", color="#cccccc", fontsize=9)
        ax_dist.set_title("Terminal Price Distribution", color="#cccccc",
                          fontsize=9, pad=6)
        ax_dist.tick_params(colors="#aaaaaa", labelsize=8)
        ax_dist.spines[:].set_color("#333344")
        ax_dist.grid(color="#333344", linewidth=0.4, linestyle=":")
        ax_dist.legend(fontsize=8, labelcolor="#cccccc",
                       facecolor="#0f1117", edgecolor="#444455",
                       framealpha=0.85)

        fig.tight_layout(pad=1.8)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Chart saved → {save_path}")
        return save_path


# ════════════════════════════════════════════════════════════════════════════
# Comparison utility — QuantLib vs NumPy implementation
# ════════════════════════════════════════════════════════════════════════════

def compare_implementations(
    ql_paths: np.ndarray,
    np_paths: np.ndarray,
    date_grid: list[date],
    save_path: str = "ou_comparison.png",
) -> None:
    """
    Plot mean ± 1-sigma bands for both implementations on the same axes
    to verify they agree in distribution.

    Parameters
    ----------
    ql_paths  : paths from LognormalOUQuantLib.simulate()
    np_paths  : paths from LognormalOUProcess.simulate()  (gas_storage_mc.py)
    date_grid : shared daily date grid
    save_path : output file path
    """
    dates_dt = [pd.Timestamp(d) for d in date_grid]

    def _band(paths):
        mean  = paths.mean(axis=0)
        upper = np.percentile(paths, 84, axis=0)   # ~mean + 1σ for lognormal
        lower = np.percentile(paths, 16, axis=0)   # ~mean - 1σ
        return mean, upper, lower

    ql_mean, ql_hi, ql_lo = _band(ql_paths)
    np_mean, np_hi, np_lo = _band(np_paths)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # QuantLib band
    ax.fill_between(dates_dt, ql_lo, ql_hi,
                    color="#3498db", alpha=0.20, label="QuantLib P16–P84")
    ax.plot(dates_dt, ql_mean, color="#3498db", lw=2.0, label="QuantLib mean")

    # NumPy band
    ax.fill_between(dates_dt, np_lo, np_hi,
                    color="#e67e22", alpha=0.20, label="NumPy P16–P84")
    ax.plot(dates_dt, np_mean, color="#e67e22", lw=2.0,
            linestyle="--", label="NumPy mean")

    ax.set_ylabel("Price (EUR/MWh)", color="#cccccc", fontsize=10)
    ax.set_title(
        "OU Process Comparison — QuantLib vs NumPy Implementation\n"
        "Mean ± 1σ bands (P16–P84)",
        color="#e0e0e0", fontsize=11, pad=10,
    )
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333344")
    ax.grid(color="#333344", linewidth=0.4, linestyle=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9, labelcolor="#cccccc",
              facecolor="#0f1117", edgecolor="#444455", framealpha=0.85)

    fig.tight_layout(pad=1.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Comparison chart saved → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Example run
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import config as cfg
    from gas_storage_mc import MarketParams, SimulationParams, LognormalOUProcess, _build_date_grid

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  QuantLib OU Simulation — Standalone Demo            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    START = cfg.CALENDAR["start_date"]
    END   = cfg.CALENDAR["end_date"]

    # ── 1. QuantLib pseudo-random ────────────────────────────────────────────
    print("[ 1 / 3 ]  QuantLib — Mersenne Twister")
    ql_sim = LognormalOUQuantLib(
        spot_price    = cfg.MARKET["spot_price"],
        kappa         = cfg.MARKET["kappa"],
        sigma         = cfg.MARKET["sigma"],
        forward_curve = cfg.MARKET["forward_curve"],
        theta         = cfg.MARKET["theta"],
        start         = START,
        end           = END,
        n_paths       = cfg.SIMULATION["n_paths"],
        seed          = cfg.SIMULATION["seed"],
        use_sobol     = False,
    )
    ql_paths = ql_sim.simulate()
    ql_sim.print_statistics(ql_paths)
    ql_sim.plot(ql_paths, save_path="ou_ql_paths.png")

    # ── 2. QuantLib Sobol ────────────────────────────────────────────────────
    print("[ 2 / 3 ]  QuantLib — Sobol quasi-random")
    ql_sobol = LognormalOUQuantLib(
        spot_price    = cfg.MARKET["spot_price"],
        kappa         = cfg.MARKET["kappa"],
        sigma         = cfg.MARKET["sigma"],
        forward_curve = cfg.MARKET["forward_curve"],
        theta         = cfg.MARKET["theta"],
        start         = START,
        end           = END,
        n_paths       = cfg.SIMULATION["n_paths"],
        seed          = cfg.SIMULATION["seed"],
        use_sobol     = True,
    )
    ql_sobol_paths = ql_sobol.simulate()
    ql_sobol.print_statistics(ql_sobol_paths)

    # ── 3. NumPy reference (from gas_storage_mc.py) ──────────────────────────
    print("[ 3 / 3 ]  NumPy reference implementation (gas_storage_mc.py)")
    market = MarketParams(
        spot_price    = cfg.MARKET["spot_price"],
        kappa         = cfg.MARKET["kappa"],
        theta         = cfg.MARKET["theta"],
        sigma         = cfg.MARKET["sigma"],
        risk_free_rate= cfg.MARKET["risk_free_rate"],
        forward_curve = cfg.MARKET["forward_curve"],
    )
    sim_p = SimulationParams(
        n_paths   = cfg.SIMULATION["n_paths"],
        seed      = cfg.SIMULATION["seed"],
        antithetic= cfg.SIMULATION["antithetic"],
    )
    date_grid = _build_date_grid(START, END)
    np_process = LognormalOUProcess(market, sim_p)
    np_paths = np_process.simulate(date_grid)
    print(f"NumPy simulation complete — {sim_p.n_paths:,} paths")

    # ── 4. Comparison ────────────────────────────────────────────────────────
    compare_implementations(ql_paths, np_paths, date_grid,
                            save_path="ou_comparison.png")

    # ── 5. Side-by-side terminal price stats ─────────────────────────────────
    print()
    print("=" * 58)
    print(f"  {'Statistic':<18} {'QuantLib MT':>12} {'QuantLib Sobol':>14} {'NumPy':>10}")
    print("=" * 58)
    for label, paths in [("QuantLib MT", ql_paths),
                         ("QuantLib Sobol", ql_sobol_paths),
                         ("NumPy", np_paths)]:
        t = paths[:, -1]
    for stat, fn in [("Mean", np.mean), ("Std", np.std),
                     ("P5",  lambda x: np.percentile(x, 5)),
                     ("P50", np.median),
                     ("P95", lambda x: np.percentile(x, 95))]:
        vals = [fn(p[:, -1]) for p in [ql_paths, ql_sobol_paths, np_paths]]
        print(f"  {stat:<18} {vals[0]:>12.4f} {vals[1]:>14.4f} {vals[2]:>10.4f}")
    print("=" * 58)
    print()
    print("Done. Output files: ou_ql_paths.png  ou_comparison.png")
