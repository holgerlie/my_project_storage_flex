"""
run_simulation.py — Gas Storage Monte Carlo: Example Run
=========================================================
Loads parameters from config.py, runs the Monte Carlo simulation,
prints a results summary, and saves:

  • storage_npv.png        — NPV distribution histogram + sample paths
  • sample_paths.csv       — 20 simulated TTF price paths (daily)
  • sample_inventory.csv   — corresponding inventory profiles

Run:
    python run_simulation.py

Requirements:
    pip install numpy pandas scipy matplotlib tqdm
    pip install QuantLib          # optional but recommended

Optional — sensitivity analysis:
    python run_simulation.py --greeks
"""

import sys
import os
import time
from datetime import date

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

import config_simple as cfg
from gas_storage_mc import (
    StorageParams,
    MarketParams,
    SimulationParams,
    OptimiserParams,
    GasStorageSimulator,
)


# ── Build parameter objects from config ────────────────────────────────────

def build_params():
    storage = StorageParams(
        min_inventory=cfg.STORAGE["min_inventory"],
        max_inventory=cfg.STORAGE["max_inventory"],
        initial_inventory=cfg.STORAGE["initial_inventory"],
        terminal_min_inventory=cfg.STORAGE["terminal_min_inventory"],
        terminal_max_inventory=cfg.STORAGE["terminal_max_inventory"],
        injection_rate_schedule={
            tuple(k): v
            for k, v in cfg.STORAGE["injection_rate_schedule"].items()
        },
        withdrawal_rate_schedule={
            tuple(k): v
            for k, v in cfg.STORAGE["withdrawal_rate_schedule"].items()
        },
        injection_efficiency=cfg.STORAGE["injection_efficiency"],
        withdrawal_efficiency=cfg.STORAGE["withdrawal_efficiency"],
        injection_cost_per_mwh=cfg.STORAGE["injection_cost_per_mwh"],
        withdrawal_cost_per_mwh=cfg.STORAGE["withdrawal_cost_per_mwh"],
        daily_fixed_cost=cfg.STORAGE["daily_fixed_cost"],
    )

    market = MarketParams(
        spot_price=cfg.MARKET["spot_price"],
        kappa=cfg.MARKET["kappa"],
        theta=cfg.MARKET["theta"],
        sigma=cfg.MARKET["sigma"],
        risk_free_rate=cfg.MARKET["risk_free_rate"],
        forward_curve=cfg.MARKET["forward_curve"],
    )

    sim_params = SimulationParams(
        n_paths=cfg.SIMULATION["n_paths"],
        seed=cfg.SIMULATION["seed"],
        antithetic=cfg.SIMULATION["antithetic"],
        n_workers=cfg.SIMULATION["n_workers"],
    )

    opt_params = OptimiserParams(
        strategy=cfg.OPTIMISER["strategy"],
        inject_threshold=cfg.OPTIMISER["inject_threshold"],
        withdraw_threshold=cfg.OPTIMISER["withdraw_threshold"],
        forward_lookback_days=cfg.OPTIMISER["forward_lookback_days"],
    )

    start = cfg.CALENDAR["start_date"]
    end   = cfg.CALENDAR["end_date"]

    return storage, market, sim_params, opt_params, start, end


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    run_greeks = "--greeks" in sys.argv

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Gas Storage Monte Carlo Valuation  (TTF / EUR)     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    storage, market, sim_params, opt_params, start, end = build_params()

    simulator = GasStorageSimulator(
        storage=storage,
        market=market,
        sim=sim_params,
        opt=opt_params,
        start=start,
        end=end,
    )

    t0 = time.perf_counter()
    daily_fwd_curve = market.print_forward_curve(start=date(2026, 4, 17), end=date(2027, 4, 17))
    results = simulator.run()
    elapsed = time.perf_counter() - t0
    print(f"Runtime: {elapsed:.1f}s\n")

    # ── Print summary ───────────────────────────────────────────────────────
    print(results.summary(percentiles=cfg.OUTPUT["percentiles"]))

    # ── Additional statistics ───────────────────────────────────────────────
    npvs = results.npvs
    prob_positive = (npvs > 0).mean()
    print(f"\n  Prob(NPV > 0)     : {prob_positive:.1%}")
    print(f"  Sharpe-like ratio : {results.mean_npv / results.std_npv:.3f}"
          "  (mean/std, undiscounted)")
    print()

    # ── Output directory ────────────────────────────────────────────────────
    out_dir = cfg.OUTPUT.get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    histogram_path = os.path.join(out_dir, "storage_npv.png")
    results.plot_npv_histogram(
        save_path=histogram_path,
        bins=cfg.OUTPUT["histogram_bins"],
        percentiles=cfg.OUTPUT["percentiles"],
    )

    if cfg.OUTPUT.get("save_path_sample", True):
        paths_csv = os.path.join(out_dir, "sample_paths.csv")
        inv_csv   = os.path.join(out_dir, "sample_inventory.csv")
        results.save_sample_paths_csv(paths_csv)
        results.save_inventory_csv(inv_csv)

    # ── Save intrinsic dispatch schedule ──────────────────────────────
    if results.intrinsic_result is not None:
        intrinsic_csv = os.path.join(out_dir, "intrinsic_dispatch.csv")
        results.intrinsic_result.save_csv(intrinsic_csv)
        print()
        print(results.intrinsic_result.summary())

    # ── Optional Greeks ────────────────────────────────────────────────────
    if run_greeks:
        print()
        print("Computing sensitivities (bump-and-revalue × 3)...")
        print("This will run 6 additional simulations — please wait.\n")

        t1 = time.perf_counter()
        delta  = simulator.delta(bump=0.50)
        vega   = simulator.vega(bump=0.01)
        theta_s = simulator.theta_sensitivity(bump=1.0)
        elapsed_g = time.perf_counter() - t1

        print(f"  Delta  (dNPV / dS0, bump=0.50 EUR/MWh)  : EUR {delta:>12,.0f} / (EUR/MWh)")
        print(f"  Vega   (dNPV / dσ,  bump=1 vol pt)      : EUR {vega:>12,.0f} / 1%")
        print(f"  θ-sens (dNPV / dθ,  bump=1 EUR/MWh)     : EUR {theta_s:>12,.0f} / (EUR/MWh)")
        print(f"  Greeks runtime: {elapsed_g:.1f}s")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
