"""Multi-run benchmarking utilities for fair variant comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .benchmarks import BenchmarkSpec
from .hill_climbing import OptimizationResult, adaptive_momentum_hill_climbing, simple_hill_climbing
from .neighborhoods import BoundaryMode, NEIGHBORHOOD_FACTORIES
from .termination import TerminationCriterion


def run_single_comparison(
    *,
    spec: BenchmarkSpec,
    initial_solution: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    neighborhood_name: str,
    boundary_mode: BoundaryMode,
    step_size: float,
    termination: TerminationCriterion,
    seed: int,
    beta1: float = 0.7,
    beta2: float = 0.9,
    momentum_gain: float = 0.75,
    reject_momentum_decay: float = 0.5,
    grow_factor: float = 1.25,
    shrink_factor: float = 0.75,
    accept_streak_to_grow: int = 2,
    reject_streak_to_shrink: int = 3,
    min_step_size: float = 1e-6,
) -> tuple[OptimizationResult, OptimizationResult]:
    """Run Simple HC and Adaptive Momentum HC from the exact same initial point."""
    factory = NEIGHBORHOOD_FACTORIES[neighborhood_name]
    neighborhood_fn = factory(boundary_mode)

    simple_res = simple_hill_climbing(
        initial_solution=initial_solution,
        objective_fn=spec.fn,
        neighborhood_fn=neighborhood_fn,
        termination=termination,
        step_size=step_size,
        bounds=bounds,
        minimize=True,
        seed=seed,
    )

    adaptive_res = adaptive_momentum_hill_climbing(
        initial_solution=initial_solution,
        objective_fn=spec.fn,
        neighborhood_fn=neighborhood_fn,
        termination=termination,
        initial_step_size=step_size,
        beta1=beta1,
        beta2=beta2,
        momentum_gain=momentum_gain,
        reject_momentum_decay=reject_momentum_decay,
        grow_factor=grow_factor,
        shrink_factor=shrink_factor,
        accept_streak_to_grow=accept_streak_to_grow,
        reject_streak_to_shrink=reject_streak_to_shrink,
        min_step_size=min_step_size,
        bounds=bounds,
        minimize=True,
        seed=seed,
    )
    return simple_res, adaptive_res


def run_batch_benchmark(
    *,
    spec: BenchmarkSpec,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    n_runs: int,
    neighborhood_name: str,
    boundary_mode: BoundaryMode,
    step_size: float,
    termination: TerminationCriterion,
    base_seed: int = 42,
    success_tol: float = 1e-2,
    adaptive_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired multi-start benchmarks across `n_runs` random initial candidates."""
    rng = np.random.default_rng(base_seed)
    low, high = bounds
    kwargs = adaptive_kwargs or {}

    rows: list[dict[str, Any]] = []
    for run_idx in range(n_runs):
        x0 = rng.uniform(low, high)
        run_seed = int(base_seed + 1000 + run_idx)
        simple_res, adaptive_res = run_single_comparison(
            spec=spec,
            initial_solution=x0,
            bounds=bounds,
            neighborhood_name=neighborhood_name,
            boundary_mode=boundary_mode,
            step_size=step_size,
            termination=termination,
            seed=run_seed,
            **kwargs,
        )

        for res in (simple_res, adaptive_res):
            gap = abs(res.best_f - spec.global_min_f)
            rows.append(
                {
                    "run": run_idx + 1,
                    "algorithm": res.algorithm_name,
                    "initial_f": res.initial_f,
                    "best_f": res.best_f,
                    "optimality_gap": gap,
                    "success": gap <= success_tol,
                    "eval_count": res.eval_count,
                    "accepted_count": res.accepted_count,
                    "accept_rate": res.accepted_count / max(1, res.eval_count),
                    "stop_reason": res.stop_reason,
                }
            )

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby("algorithm", as_index=False)
        .agg(
            runs=("run", "count"),
            success_rate_pct=("success", lambda s: 100.0 * float(np.mean(s))),
            mean_best_f=("best_f", "mean"),
            median_best_f=("best_f", "median"),
            std_best_f=("best_f", "std"),
            min_best_f=("best_f", "min"),
            max_best_f=("best_f", "max"),
            mean_evals=("eval_count", "mean"),
            mean_accept_rate_pct=("accept_rate", lambda s: 100.0 * float(np.mean(s))),
        )
    )
    return runs_df, summary_df
