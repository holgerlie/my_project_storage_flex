"""
processes.py — Stochastic Price Process Classes for Gas Storage Valuation
=========================================================================
Three process classes for Monte Carlo path generation, all returning a
PathBundle so they are drop-in interchangeable in GasStorageSimulator and
the LSMC engine.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Class                  Backend       Scheme        state2            │
    ├──────────────────────────────────────────────────────────────────────┤
    │ LognormalOUProcess     NumPy         Exact analytic None             │
    │ TwoFactorSchwartzProcess NumPy       Exact analytic χ_t log-devn     │
    │ HestonProcess          QuantLib      QE Martingale  v_t variance     │
    └──────────────────────────────────────────────────────────────────────┘

Design principles
-----------------
1. Exact discretisation wherever the SDE has a closed-form transition
   density.  No Euler bias.  No time-step refinement needed.

2. Full vectorisation.  All paths are generated in a single NumPy operation
   (one RNG draw of shape (n_paths, n_steps)).  No Python loops over paths.

3. Antithetic variates.  Controlled by the SimulationParams.antithetic flag.
   The antithetic mirror is constructed in the RNG layer, not the process
   layer — identical to the existing LognormalOUProcess in gas_storage_mc.py.

4. QuantLib is used only for HestonProcess, where the Quadratic Exponential
   (QE) Martingale discretisation (Andersen 2008) is non-trivial to implement
   correctly.  The QE scheme handles the zero-boundary of the variance process
   without reflection or truncation and preserves the martingale property of
   the discounted spot price.  This is the desk standard for Heston.

5. PathBundle is the sole output type.  The LSMC regression layer reads
   bundle.model to select basis functions; it never inspects the process class.

Exact discretisation derivations
---------------------------------
LognormalOU (single-factor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  d(ln S) = κ(ln F(t) − ln S) dt + σ dW

  Exact transition (conditional on F(t), dt):

    ln S_{t+dt} = ln F(t+dt)
                  + (ln S_t − ln F(t+dt)) · exp(−κ dt)
                  + σ · sqrt((1 − exp(−2κ dt)) / (2κ)) · Z

  where Z ~ N(0,1), independent across steps.

  The conditional mean is ln F(t+dt) + (ln S_t − ln F(t+dt))·exp(−κ dt),
  so the process correctly tracks the forward curve pillar-by-pillar.

  Variance: σ²·(1 − exp(−2κ dt))/(2κ) — the OU asymptotic variance scaled
  by the mean-reversion decay over the step.

Two-Factor Schwartz-Smith (2000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ln S_t = χ_t + η_t

  dχ_t = −κ_χ · χ_t · dt  +  σ_χ · dW₁       (short-term, mean-reverts to 0)
  dη_t =  μ_η             · dt  +  σ_η · dW₂   (long-term, arithmetic BM with drift)

  Corr(dW₁, dW₂) = ρ

  Exact transition densities (both factors are affine SDEs):

    χ_{t+dt} = χ_t · exp(−κ_χ dt)
               + σ_χ · sqrt((1 − exp(−2κ_χ dt)) / (2κ_χ)) · Z₁

    η_{t+dt} = η_t + μ_η · dt
               + σ_η · sqrt(dt) · Z₂

  Correlated draw: [Z₁, Z₂] ~ N(0, Σ) where Σ = [[1, ρ], [ρ, 1]].
  Cholesky decomposition L such that L·L' = Σ applied to independent standard
  normals gives the correlated pair:

    Z₁ = ε₁
    Z₂ = ρ · ε₁ + sqrt(1 − ρ²) · ε₂,   ε₁,ε₂ ~ N(0,1) i.i.d.

  The spot price is S_t = exp(χ_t + η_t).  state2 carries χ_t because it is
  the fast mean-reverting factor that drives seasonal spread optionality and
  is the primary LSMC regression variable.

Heston (1993) — QuantLib QE scheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  dS_t  = r · S_t · dt  +  sqrt(v_t) · S_t · dW₁
  dv_t  = κ · (θ − v_t) · dt  +  ξ · sqrt(v_t) · dW₂
  Corr(dW₁, dW₂) = ρ

  The QE scheme (Andersen 2008) is used because:
  • The variance SDE has no closed-form transition density.
  • Euler and Milstein schemes produce negative variance at low v_t,
    requiring ad-hoc fixes (full truncation, reflection) that introduce bias.
  • QE uses a moment-matched piecewise approximation of the non-central chi-
    squared transition density, switching between a quadratic and exponential
    fit based on the ratio v_t / θ.  It is bias-free at zero and preserves
    the martingale property.

  This is the recommended discretisation in Glasserman & Kim (2011) and
  is the desk standard for Heston MC valuation.

Dependencies
------------
    NumPy  — LognormalOUProcess, TwoFactorSchwartzProcess
    QuantLib — HestonProcess only
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# PathBundle — shared output contract
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PathBundle:
    """
    Simulation output container shared by all process classes.

    Attributes
    ----------
    spots  : np.ndarray, shape (n_paths, n_steps)
             Simulated spot prices in EUR/MWh.
             Column 0 is always S0, identical across all paths.

    state2 : np.ndarray or None, shape (n_paths, n_steps)
             Second factor — only present for multi-factor models:
               TwoFactorSchwartzProcess : χ_t  (short-term log-deviation)
               HestonProcess            : v_t  (instantaneous variance)
             None for LognormalOUProcess.

    model  : str
             Process identifier consumed by BasisFunctions:
             "lognormal_ou" | "schwartz_smith" | "heston"
    """
    spots:  np.ndarray
    state2: Optional[np.ndarray]
    model:  str


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _antithetic_normals(
    rng:       np.random.Generator,
    n_paths:   int,
    n_steps:   int,
    antithetic: bool,
) -> np.ndarray:
    """
    Draw standard normals of shape (n_paths, n_steps).

    If antithetic=True, the first n_paths//2 rows are independent draws and
    the remaining rows are their negatives.  n_paths must be even.
    """
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        n_ind = n_paths // 2
        Z_ind = rng.standard_normal((n_ind, n_steps))
        return np.vstack([Z_ind, -Z_ind])
    return rng.standard_normal((n_paths, n_steps))


def _dt_array(date_grid: List[date]) -> np.ndarray:
    """
    Return step sizes dt[i] = (date_grid[i+1] - date_grid[i]).days / 365.25
    as a 1-D array of length n_steps = len(date_grid) - 1.
    """
    return np.array(
        [(date_grid[i + 1] - date_grid[i]).days / 365.25
         for i in range(len(date_grid) - 1)]
    )


# ════════════════════════════════════════════════════════════════════════════
# Process 1 — LognormalOUProcess  (exact analytic, fully vectorised)
# ════════════════════════════════════════════════════════════════════════════

class LognormalOUProcess:
    """
    Single-factor lognormal OU process.  Exact analytical discretisation.

    Log-prices follow an additive OU around the log-forward curve:

        d(ln S) = κ · (ln F(t) − ln S) · dt  +  σ · dW

    Exact transition (no Euler bias):

        ln S_{t+dt} = ln F(t+dt)
                      + (ln S_t − ln F(t+dt)) · exp(−κ dt)
                      + σ · sqrt((1 − exp(−2κ dt)) / (2κ)) · Z,   Z ~ N(0,1)

    Implementation
    --------------
    All n_paths paths are evolved simultaneously using broadcasting.
    The forward curve is pre-evaluated on the full date grid before the
    time-stepping loop — one interpolation pass, no per-path overhead.

    This class replaces the LognormalOUProcess in gas_storage_mc.py.
    It accepts the same fwd_func callable (MarketParams.forward_price) so
    it is a drop-in replacement.

    Parameters
    ----------
    spot_price  : float    — S0 (EUR/MWh)
    kappa       : float    — mean-reversion speed (yr⁻¹)
    sigma       : float    — log-price annual volatility
    fwd_func    : callable — date → EUR/MWh forward price
    n_paths     : int
    seed        : int or None
    antithetic  : bool     — antithetic variates (n_paths must be even)
    """

    MODEL = "lognormal_ou"

    def __init__(
        self,
        spot_price : float,
        kappa      : float,
        sigma      : float,
        fwd_func,
        n_paths    : int  = 10_000,
        seed       : Optional[int] = 42,
        antithetic : bool = True,
    ):
        self.spot_price = spot_price
        self.kappa      = kappa
        self.sigma      = sigma
        self.fwd_func   = fwd_func
        self.n_paths    = n_paths
        self.seed       = seed
        self.antithetic = antithetic

    def simulate(self, date_grid: List[date]) -> PathBundle:
        """
        Returns PathBundle(spots=(n_paths, n_steps), state2=None, model="lognormal_ou").
        """
        n_steps = len(date_grid) - 1
        rng     = np.random.default_rng(self.seed)

        # Pre-compute log-forward curve on full date grid  (n_steps+1,)
        log_fwd = np.log(np.array([self.fwd_func(d) for d in date_grid]))

        # Step sizes and exact OU coefficients  (n_steps,)
        dt      = _dt_array(date_grid)
        e       = np.exp(-self.kappa * dt)
        std     = self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.kappa * dt))
                                       / (2.0 * self.kappa))

        # Correlated normals  (n_paths, n_steps)
        Z = _antithetic_normals(rng, self.n_paths, n_steps, self.antithetic)

        # Vectorised path evolution
        log_paths       = np.empty((self.n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(self.spot_price)

        for i in range(n_steps):
            log_paths[:, i + 1] = (
                log_fwd[i + 1]
                + (log_paths[:, i] - log_fwd[i + 1]) * e[i]
                + std[i] * Z[:, i]
            )

        return PathBundle(
            spots  = np.exp(log_paths),
            state2 = None,
            model  = self.MODEL,
        )


# ════════════════════════════════════════════════════════════════════════════
# Process 2 — TwoFactorSchwartzProcess  (exact analytic, fully vectorised)
# ════════════════════════════════════════════════════════════════════════════

class TwoFactorSchwartzProcess:
    """
    Two-factor Schwartz-Smith (2000) model.  Exact analytical discretisation.

    The spot price decomposes as:

        ln S_t = χ_t + η_t

    where χ_t is a fast mean-reverting short-term factor (seasonal spreads)
    and η_t is a near-random-walk long-term equilibrium (structural level).

    SDEs:
        dχ_t = −κ_χ · χ_t · dt  +  σ_χ · dW₁
        dη_t =  μ_η             · dt  +  σ_η · dW₂
        Corr(dW₁, dW₂) = ρ

    Exact transitions:
        χ_{t+dt} = χ_t · exp(−κ_χ dt)
                   + σ_χ · sqrt((1 − exp(−2κ_χ dt)) / (2κ_χ)) · Z₁

        η_{t+dt} = η_t + μ_η · dt
                   + σ_η · sqrt(dt) · Z₂

    Correlated draw via Cholesky:
        Z₁ = ε₁
        Z₂ = ρ · ε₁ + sqrt(1 − ρ²) · ε₂,    ε₁, ε₂ ~ N(0,1) i.i.d.

    Relevance for gas storage
    -------------------------
    This is the natural model for seasonal storage:
      • χ_t captures the summer/winter spread that drives inject/withdraw
        decisions.  Fast mean-reversion (κ_χ ≈ 2–5 yr⁻¹) implies a half-life
        of 2–4 months, consistent with observed TTF seasonality.
      • η_t captures the slow structural level (LNG import capacity, demand
        growth).  Setting μ_η ≈ 0 makes it a martingale — no P-measure drift
        enters the risk-neutral valuation.
      • The correlation ρ between factors is typically small (|ρ| < 0.3 for
        TTF), reflecting that seasonal and structural shocks are largely
        independent.

    Initial state decomposition
    ---------------------------
    Given spot S0 and a long-term forward price F_LT (last pillar):
        η0 = ln(F_LT)                  — long-run equilibrium in log-space
        χ0 = ln(S0) − η0              — current short-term deviation

    Parameters
    ----------
    spot_price  : float    — S0 (EUR/MWh)
    eta_0       : float    — long-term equilibrium level F_LT (EUR/MWh)
    kappa_xi    : float    — short-term mean-reversion speed (yr⁻¹), e.g. 2.0
    sigma_xi    : float    — short-term factor volatility, e.g. 0.35
    mu_eta      : float    — long-term drift (risk-neutral, typically 0.0)
    sigma_eta   : float    — long-term factor volatility, e.g. 0.15
    rho         : float    — factor correlation, e.g. −0.20
    n_paths     : int
    seed        : int or None
    antithetic  : bool
    """

    MODEL = "schwartz_smith"

    def __init__(
        self,
        spot_price : float,
        eta_0      : float,
        kappa_xi   : float,
        sigma_xi   : float,
        mu_eta     : float,
        sigma_eta  : float,
        rho        : float,
        n_paths    : int  = 10_000,
        seed       : Optional[int] = 42,
        antithetic : bool = True,
    ):
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must be in (−1, 1), got {rho}")

        self.spot_price = spot_price
        self.kappa_xi   = kappa_xi
        self.sigma_xi   = sigma_xi
        self.mu_eta     = mu_eta
        self.sigma_eta  = sigma_eta
        self.rho        = rho
        self.n_paths    = n_paths
        self.seed       = seed
        self.antithetic = antithetic

        # Initial state decomposition
        self.eta0 = np.log(eta_0)               # long-term log-level
        self.xi0  = np.log(spot_price) - self.eta0   # short-term log-deviation

    def simulate(self, date_grid: List[date]) -> PathBundle:
        """
        Returns PathBundle:
          spots  : (n_paths, n_steps)  — S_t = exp(χ_t + η_t)
          state2 : (n_paths, n_steps)  — χ_t paths (short-term log-deviation)
          model  : "schwartz_smith"
        """
        n_steps = len(date_grid) - 1
        rng     = np.random.default_rng(self.seed)

        # Step sizes  (n_steps,)
        dt = _dt_array(date_grid)

        # Exact OU coefficients for χ  (n_steps,)
        e_xi   = np.exp(-self.kappa_xi * dt)
        std_xi = self.sigma_xi * np.sqrt(
            (1.0 - np.exp(-2.0 * self.kappa_xi * dt)) / (2.0 * self.kappa_xi)
        )

        # BM increments for η  (n_steps,)
        std_eta = self.sigma_eta * np.sqrt(dt)

        # Independent standard normals  (n_paths, n_steps, 2)
        # Drawing both factors at once in one RNG call is the most efficient
        # approach; antithetic mirroring applies to the full 2-D draw.
        eps = rng.standard_normal((self.n_paths, n_steps, 2))
        if self.antithetic:
            if self.n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True.")
            n_ind = self.n_paths // 2
            eps[:n_ind]  =  eps[:n_ind]      # independent half (unchanged)
            eps[n_ind:]  = -eps[:n_ind]      # antithetic mirror

        # Cholesky decomposition for correlation ρ
        # Z₁ = ε₁,   Z₂ = ρ·ε₁ + sqrt(1−ρ²)·ε₂
        rho_perp = np.sqrt(1.0 - self.rho ** 2)
        Z1 = eps[:, :, 0]                                     # (n_paths, n_steps)
        Z2 = self.rho * eps[:, :, 0] + rho_perp * eps[:, :, 1]  # (n_paths, n_steps)

        # Evolve both factors in vectorised loops
        xi  = np.empty((self.n_paths, n_steps + 1))
        eta = np.empty((self.n_paths, n_steps + 1))
        xi[:, 0]  = self.xi0
        eta[:, 0] = self.eta0

        for i in range(n_steps):
            xi[:, i + 1]  = xi[:, i] * e_xi[i]  +  std_xi[i]  * Z1[:, i]
            eta[:, i + 1] = eta[:, i] + self.mu_eta * dt[i]  +  std_eta[i] * Z2[:, i]

        return PathBundle(
            spots  = np.exp(xi + eta),
            state2 = xi,
            model  = self.MODEL,
        )


# ════════════════════════════════════════════════════════════════════════════
# Process 3 — HestonProcess  (QuantLib QE Martingale, path loop unavoidable)
# ════════════════════════════════════════════════════════════════════════════

class HestonProcess:
    """
    Heston (1993) stochastic volatility model.
    QuantLib Quadratic Exponential (QE) Martingale discretisation.

    SDEs:
        dS_t  =  r · S_t · dt  +  sqrt(v_t) · S_t · dW₁
        dv_t  =  κ · (θ_v − v_t) · dt  +  ξ · sqrt(v_t) · dW₂
        Corr(dW₁, dW₂) = ρ

    Why QuantLib here
    -----------------
    The CIR variance process (v_t) has no analytically tractable transition
    density for the joint (S_t, v_t) system.  Euler and Milstein schemes
    produce negative variance near zero, requiring truncation or reflection
    that introduces systematic bias.

    The QE scheme (Andersen 2008) approximates the non-central chi-squared
    transition of v_t by moment-matching a two-parameter family that switches
    between a quadratic and exponential form based on ψ = Var(v_t)/E(v_t)².
    It is numerically bias-free at v_t = 0, unconditionally stable, and
    preserves the E[S_T] = S_0 · e^{rT} martingale property.

    The QuantLib Python path loop (one path at a time) is unavoidable because
    the QE scheme requires the current variance v_t to determine the regime
    switch — this is inherently sequential per-path.  For production use with
    n_paths > 50,000, consider the QuantLib C++ extension or a Cython/Numba
    reimplementation of the QE scheme.

    Feller condition
    ----------------
    2κθ_v > ξ² guarantees v_t > 0 almost surely.  If violated, a warning is
    issued and the QE scheme handles it gracefully via the exponential branch.

    Parameters
    ----------
    spot_price      : float    — S0 (EUR/MWh)
    risk_free_rate  : float    — continuous risk-free rate r
    v0              : float    — initial variance (σ²), e.g. 0.09 for σ=30%
    kappa           : float    — variance mean-reversion speed (yr⁻¹)
    theta_v         : float    — long-run variance
    xi              : float    — vol-of-vol
    rho             : float    — correlation dW₁·dW₂ (typically negative)
    n_paths         : int
    seed            : int or None
    use_sobol       : bool     — Sobol quasi-random (lower discrepancy)
    """

    MODEL = "heston"

    def __init__(
        self,
        spot_price     : float,
        risk_free_rate : float,
        v0             : float,
        kappa          : float,
        theta_v        : float,
        xi             : float,
        rho            : float,
        n_paths        : int  = 10_000,
        seed           : Optional[int] = 42,
        use_sobol      : bool = False,
    ):
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must be in (−1, 1), got {rho}")
        if v0 <= 0:
            raise ValueError(f"v0 must be positive, got {v0}")

        feller = 2.0 * kappa * theta_v - xi ** 2
        if feller <= 0:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2κθ − ξ² = {feller:.4f} ≤ 0. "
                "Variance can reach zero.  QE scheme handles this, but "
                "review calibration.",
                stacklevel=2,
            )

        self.spot_price     = spot_price
        self.risk_free_rate = risk_free_rate
        self.v0             = v0
        self.kappa          = kappa
        self.theta_v        = theta_v
        self.xi             = xi
        self.rho            = rho
        self.n_paths        = n_paths
        self.seed           = seed
        self.use_sobol      = use_sobol

    def simulate(self, date_grid: List[date]) -> PathBundle:
        """
        Returns PathBundle:
          spots  : (n_paths, n_steps)  — S_t
          state2 : (n_paths, n_steps)  — v_t (instantaneous variance)
          model  : "heston"
        """
        try:
            import QuantLib as ql
        except ImportError:
            raise ImportError(
                "QuantLib is required for HestonProcess.  "
                "Install with: pip install QuantLib"
            )

        n_steps  = len(date_grid) - 1
        ref_date = date_grid[0]
        T        = (date_grid[-1] - ref_date).days / 365.25

        # ── QuantLib setup ───────────────────────────────────────────────────
        ql.Settings.instance().evaluationDate = ql.Date(
            ref_date.day, ref_date.month, ref_date.year
        )
        dc  = ql.Actual365Fixed()
        cal = ql.NullCalendar()

        def _flat_ts(rate: float) -> ql.YieldTermStructureHandle:
            return ql.YieldTermStructureHandle(
                ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(rate)), dc)
            )

        process = ql.HestonProcess(
            _flat_ts(self.risk_free_rate),   # risk-free
            _flat_ts(0.0),                   # dividend / convenience yield
            ql.QuoteHandle(ql.SimpleQuote(self.spot_price)),
            self.v0,
            self.kappa,
            self.theta_v,
            self.xi,
            self.rho,
            ql.HestonProcess.QuadraticExponentialMartingale,
        )

        time_grid = ql.TimeGrid(T, n_steps)

        # ── RNG: Sobol or Mersenne Twister ───────────────────────────────────
        # n_dims = 2 (spot + variance) × n_steps
        n_dims = 2 * n_steps
        if self.use_sobol:
            uniform_rsg  = ql.UniformLowDiscrepancySequenceGenerator(n_dims)
            gaussian_rsg = ql.GaussianLowDiscrepancySequenceGenerator(uniform_rsg)
            generator    = ql.GaussianSobolMultiPathGenerator(
                process, time_grid, gaussian_rsg, False
            )
        else:
            ql_seed      = (self.seed + 1) if self.seed is not None else 42
            uniform_rsg  = ql.UniformRandomSequenceGenerator(
                n_dims, ql.UniformRandomGenerator(ql_seed)
            )
            gaussian_rsg = ql.GaussianRandomSequenceGenerator(uniform_rsg)
            generator    = ql.GaussianMultiPathGenerator(
                process, time_grid, gaussian_rsg, False
            )

        # ── Path extraction  (Python loop — unavoidable for QE scheme) ───────
        spot_paths = np.empty((self.n_paths, n_steps + 1))
        var_paths  = np.empty((self.n_paths, n_steps + 1))

        for i in range(self.n_paths):
            mp = generator.next().value()   # MultiPath
            spot_paths[i] = [mp[0][j] for j in range(n_steps + 1)]
            var_paths[i]  = [mp[1][j] for j in range(n_steps + 1)]

        return PathBundle(
            spots  = spot_paths,
            state2 = var_paths,
            model  = self.MODEL,
        )


# ════════════════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))
    import config as cfg
    from gas_storage_mc import _build_date_grid, MarketParams

    market    = MarketParams(**cfg.MARKET)
    date_grid = _build_date_grid(cfg.CALENDAR["start_date"], cfg.CALENDAR["end_date"])
    n_paths   = cfg.SIMULATION["n_paths"]
    seed      = cfg.SIMULATION["seed"]
    antithetic = cfg.SIMULATION["antithetic"]

    fwd_pillars = sorted(market.forward_curve.items())
    eta_0_price = fwd_pillars[-1][1]   # last forward pillar as long-run level

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  processes.py — Smoke Test                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  date_grid : {date_grid[0]} → {date_grid[-1]}  ({len(date_grid)} steps)")
    print(f"  n_paths   : {n_paths:,}   antithetic: {antithetic}")
    print()

    # ── 1. LognormalOUProcess ────────────────────────────────────────────────
    print("[ 1 / 3 ]  LognormalOUProcess  (exact analytic, NumPy)")
    t0 = time.perf_counter()
    proc_ou = LognormalOUProcess(
        spot_price = market.spot_price,
        kappa      = market.kappa,
        sigma      = market.sigma,
        fwd_func   = market.forward_price,
        n_paths    = n_paths,
        seed       = seed,
        antithetic = antithetic,
    )
    bundle_ou = proc_ou.simulate(date_grid)
    t_ou = time.perf_counter() - t0
    print(f"  elapsed      : {t_ou*1000:.1f} ms")
    print(f"  spots shape  : {bundle_ou.spots.shape}")
    print(f"  S0 (col 0)   : {bundle_ou.spots[:, 0].mean():.4f}  (expect {market.spot_price})")
    print(f"  mean terminal: {bundle_ou.spots[:, -1].mean():.4f}")
    print(f"  state2       : {bundle_ou.state2}  (None — single factor)")
    print()

    # ── 2. TwoFactorSchwartzProcess ──────────────────────────────────────────
    print("[ 2 / 3 ]  TwoFactorSchwartzProcess  (exact analytic, NumPy)")
    t0 = time.perf_counter()
    proc_ss = TwoFactorSchwartzProcess(
        spot_price = market.spot_price,
        eta_0      = eta_0_price,
        kappa_xi   = 2.0,
        sigma_xi   = 0.35,
        mu_eta     = 0.0,
        sigma_eta  = 0.15,
        rho        = -0.20,
        n_paths    = n_paths,
        seed       = seed,
        antithetic = antithetic,
    )
    bundle_ss = proc_ss.simulate(date_grid)
    t_ss = time.perf_counter() - t0
    print(f"  elapsed      : {t_ss*1000:.1f} ms")
    print(f"  spots shape  : {bundle_ss.spots.shape}")
    print(f"  S0 (col 0)   : {bundle_ss.spots[:, 0].mean():.4f}  (expect {market.spot_price:.4f})")
    print(f"  mean terminal: {bundle_ss.spots[:, -1].mean():.4f}")
    print(f"  χ_t shape    : {bundle_ss.state2.shape}  (short-term factor)")
    print(f"  χ mean @ end : {bundle_ss.state2[:, -1].mean():.4f}  (expect ~0)")
    print(f"  χ std  @ end : {bundle_ss.state2[:, -1].std():.4f}")
    print()

    # ── 3. HestonProcess ────────────────────────────────────────────────────
    print("[ 3 / 3 ]  HestonProcess  (QuantLib QE Martingale)")
    t0 = time.perf_counter()
    proc_h = HestonProcess(
        spot_price     = market.spot_price,
        risk_free_rate = market.risk_free_rate,
        v0             = market.sigma ** 2,
        kappa          = 2.0,
        theta_v        = market.sigma ** 2,
        xi             = 0.40,
        rho            = -0.60,
        n_paths        = n_paths,
        seed           = seed,
        use_sobol      = False,
    )
    bundle_h = proc_h.simulate(date_grid)
    t_h = time.perf_counter() - t0
    print(f"  elapsed      : {t_h*1000:.1f} ms")
    print(f"  spots shape  : {bundle_h.spots.shape}")
    print(f"  S0 (col 0)   : {bundle_h.spots[:, 0].mean():.4f}  (expect {market.spot_price})")
    print(f"  mean terminal: {bundle_h.spots[:, -1].mean():.4f}")
    print(f"  v_t shape    : {bundle_h.state2.shape}  (variance factor)")
    print(f"  v0 (col 0)   : {bundle_h.state2[:, 0].mean():.6f}  (expect {market.sigma**2:.6f})")
    print(f"  mean v @ end : {bundle_h.state2[:, -1].mean():.6f}  (expect ~{market.sigma**2:.6f})")
    print()
    print(f"  Speed summary:  OU {t_ou*1000:.0f} ms  |  "
          f"Schwartz {t_ss*1000:.0f} ms  |  "
          f"Heston {t_h*1000:.0f} ms")
    print()
    print("Smoke test complete.")
