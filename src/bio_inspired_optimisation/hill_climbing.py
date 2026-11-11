"""Reusable Hill Climbing algorithms with exact objective-evaluation tracking."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from .neighborhoods import Bounds, NeighborhoodFn, apply_bounds
from .termination import TerminationCriterion


def _is_better(candidate_f: float, current_f: float, minimize: bool) -> bool:
    return candidate_f < current_f if minimize else candidate_f > current_f


def _accepts_bias(neighborhood_fn: Callable) -> bool:
    try:
        sig = inspect.signature(neighborhood_fn)
    except (TypeError, ValueError):
        return False
    return "bias" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _propose_candidate(
    neighborhood_fn: NeighborhoodFn | Callable,
    has_bias_param: bool,
    x: np.ndarray,
    step_size: float,
    rng: np.random.Generator,
    bounds: Bounds | None,
    drift: np.ndarray | None,
) -> np.ndarray:
    if drift is None:
        return neighborhood_fn(x, step_size, rng, bounds=bounds)
    if has_bias_param:
        return neighborhood_fn(x, step_size, rng, bounds=bounds, bias=drift)
    shifted = apply_bounds(x + drift, bounds, mode="clip")
    return neighborhood_fn(shifted, step_size, rng, bounds=bounds)


class TrackedObjective:
    """Wrapper around an objective function that strictly counts every evaluation."""

    def __init__(self, objective_fn: Callable[[np.ndarray], float]) -> None:
        self._objective_fn = objective_fn
        self.eval_count = 0

    def __call__(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self._objective_fn(x))


@dataclass(frozen=True)
class EvaluationRecord:
    """Snapshot recorded at every single objective function evaluation."""

    eval_index: int
    iter_index: int
    x: np.ndarray
    f: float
    accepted: bool
    best_x: np.ndarray
    best_f: float
    step_size: float
    drift: np.ndarray | None = None
    prev_x: np.ndarray | None = None
    prev_f: float | None = None
    step_size_before: float | None = None
    offset: np.ndarray | None = None
    delta: np.ndarray | None = None
    m_before: np.ndarray | None = None
    v_before: np.ndarray | None = None
    t_mom_before: int = 0
    m_hat: np.ndarray | None = None
    v_hat: np.ndarray | None = None
    d_norm: np.ndarray | None = None
    m_after: np.ndarray | None = None
    v_after: np.ndarray | None = None
    t_mom_after: int = 0
    accept_streak_before: int = 0
    reject_streak_before: int = 0
    accept_streak_after: int = 0
    reject_streak_after: int = 0
    step_event: str = ""


@dataclass
class OptimizationResult:
    """Full trajectory and summary metrics of a Hill Climbing run."""

    algorithm_name: str
    initial_x: np.ndarray
    initial_f: float
    best_x: np.ndarray
    best_f: float
    eval_count: int
    iter_count: int
    accepted_count: int
    stop_reason: str
    history: list[EvaluationRecord]
    params: dict[str, Any] = field(default_factory=dict)

    def accepted_points(self, up_to_eval: int | None = None) -> np.ndarray:
        """Return array of accepted positions shape (k, dim) up to `up_to_eval`."""
        limit = self.eval_count if up_to_eval is None else up_to_eval
        pts = [rec.x for rec in self.history if rec.eval_index <= limit and rec.accepted]
        return np.vstack(pts) if pts else np.empty((0, self.initial_x.size), dtype=float)

    def accepted_values(self, up_to_eval: int | None = None) -> np.ndarray:
        """Return objective values at accepted positions up to `up_to_eval`."""
        limit = self.eval_count if up_to_eval is None else up_to_eval
        return np.array(
            [rec.f for rec in self.history if rec.eval_index <= limit and rec.accepted],
            dtype=float,
        )

    def rejected_points(self, up_to_eval: int | None = None) -> np.ndarray:
        """Return array of rejected probe positions shape (k, dim) up to `up_to_eval`."""
        limit = self.eval_count if up_to_eval is None else up_to_eval
        pts = [rec.x for rec in self.history if rec.eval_index <= limit and not rec.accepted]
        return np.vstack(pts) if pts else np.empty((0, self.initial_x.size), dtype=float)

    def rejected_values(self, up_to_eval: int | None = None) -> np.ndarray:
        """Return objective values at rejected probes up to `up_to_eval`."""
        limit = self.eval_count if up_to_eval is None else up_to_eval
        return np.array(
            [rec.f for rec in self.history if rec.eval_index <= limit and not rec.accepted],
            dtype=float,
        )

    def eval_indices(self, up_to_eval: int | None = None) -> np.ndarray:
        limit = self.eval_count if up_to_eval is None else up_to_eval
        return np.array(
            [rec.eval_index for rec in self.history if rec.eval_index <= limit],
            dtype=int,
        )

    def best_f_curve(self, up_to_eval: int | None = None) -> np.ndarray:
        limit = self.eval_count if up_to_eval is None else up_to_eval
        return np.array(
            [rec.best_f for rec in self.history if rec.eval_index <= limit],
            dtype=float,
        )

    def step_size_curve(self, up_to_eval: int | None = None) -> np.ndarray:
        limit = self.eval_count if up_to_eval is None else up_to_eval
        return np.array(
            [rec.step_size for rec in self.history if rec.eval_index <= limit],
            dtype=float,
        )


def _prepare_initial(initial_solution: ArrayLike, bounds: Bounds | None) -> np.ndarray:
    x0 = np.asarray(initial_solution, dtype=float).copy()
    if x0.ndim == 0:
        x0 = x0.reshape(1)
    if x0.ndim != 1:
        raise ValueError(f"initial_solution must be 1D, got shape {x0.shape}.")
    return apply_bounds(x0, bounds, mode="clip")


def simple_hill_climbing(
    initial_solution: ArrayLike,
    objective_fn: Callable[[np.ndarray], float],
    neighborhood_fn: NeighborhoodFn,
    termination: TerminationCriterion,
    *,
    step_size: float = 0.5,
    bounds: Bounds | None = None,
    minimize: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult:
    """Standard Simple Hill Climbing (single random neighbor probe per iteration)."""
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    gen = rng if rng is not None else np.random.default_rng(seed)
    tracker = TrackedObjective(objective_fn)
    x_curr = _prepare_initial(initial_solution, bounds)
    f_curr = tracker(x_curr)
    x_init = x_curr.copy()
    f_init = f_curr
    dim = x_curr.size
    sigma = float(step_size)

    history: list[EvaluationRecord] = [
        EvaluationRecord(
            eval_index=tracker.eval_count,
            iter_index=0,
            x=x_curr.copy(),
            f=f_curr,
            accepted=True,
            best_x=x_curr.copy(),
            best_f=f_curr,
            step_size=sigma,
            drift=None,
            prev_x=x_curr.copy(),
            prev_f=f_curr,
            step_size_before=sigma,
            offset=np.zeros(dim, dtype=float),
            delta=None,
            accept_streak_before=0,
            reject_streak_before=0,
            accept_streak_after=0,
            reject_streak_after=0,
            step_event="Init",
        )
    ]

    iter_count = 0
    no_improve_count = 0
    accepted_count = 1
    accept_streak = 0
    reject_streak = 0

    while True:
        reason = termination.check(
            eval_count=tracker.eval_count,
            iter_count=iter_count,
            no_improve_count=no_improve_count,
            best_f=f_curr,
            current_step_size=sigma,
            minimize=minimize,
        )
        if reason is not None:
            stop_reason = reason
            break

        iter_count += 1
        prev_x = x_curr.copy()
        prev_f = f_curr
        acc_before = accept_streak
        rej_before = reject_streak

        x_new = neighborhood_fn(prev_x, sigma, gen, bounds=bounds)
        offset = x_new - prev_x
        f_new = tracker(x_new)

        if _is_better(f_new, f_curr, minimize=minimize):
            delta = x_new - prev_x
            x_curr = x_new.copy()
            f_curr = f_new
            accepted = True
            accepted_count += 1
            no_improve_count = 0
            accept_streak += 1
            reject_streak = 0
            event = "Accept"
        else:
            delta = None
            accepted = False
            no_improve_count += 1
            accept_streak = 0
            reject_streak += 1
            event = "Reject"

        history.append(
            EvaluationRecord(
                eval_index=tracker.eval_count,
                iter_index=iter_count,
                x=x_new.copy(),
                f=f_new,
                accepted=accepted,
                best_x=x_curr.copy(),
                best_f=f_curr,
                step_size=sigma,
                drift=None,
                prev_x=prev_x,
                prev_f=prev_f,
                step_size_before=sigma,
                offset=offset,
                delta=delta,
                accept_streak_before=acc_before,
                reject_streak_before=rej_before,
                accept_streak_after=accept_streak,
                reject_streak_after=reject_streak,
                step_event=event,
            )
        )

    return OptimizationResult(
        algorithm_name="Simple Hill Climbing",
        initial_x=x_init,
        initial_f=f_init,
        best_x=x_curr.copy(),
        best_f=f_curr,
        eval_count=tracker.eval_count,
        iter_count=iter_count,
        accepted_count=accepted_count,
        stop_reason=stop_reason,
        history=history,
        params={
            "step_size": sigma,
            "minimize": minimize,
            "seed": seed,
            "max_evals": termination.max_evals,
            "max_no_improve": termination.max_no_improve,
        },
    )


def adaptive_momentum_hill_climbing(
    initial_solution: ArrayLike,
    objective_fn: Callable[[np.ndarray], float],
    neighborhood_fn: NeighborhoodFn,
    termination: TerminationCriterion,
    *,
    initial_step_size: float = 0.5,
    beta1: float = 0.7,
    beta2: float = 0.9,
    momentum_gain: float = 0.75,
    reject_momentum_decay: float = 0.5,
    grow_factor: float = 1.25,
    shrink_factor: float = 0.75,
    accept_streak_to_grow: int = 2,
    reject_streak_to_shrink: int = 3,
    min_step_size: float = 1e-6,
    max_step_size: float | None = None,
    eps: float = 1e-8,
    bounds: Bounds | None = None,
    minimize: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult:
    """Derivative-free Adaptive Momentum Hill Climbing (Adam-inspired first/second moments).

    Maintains bias-corrected exponential moving averages (m, v) over successful step
    displacements delta = x_new - x_old to bias the neighborhood search center along
    consistent downhill trends, while dynamically scaling the neighborhood radius sigma
    on streaks of accepted or rejected probes.
    """
    if initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive.")
    if not (0.0 <= beta1 < 1.0):
        raise ValueError("beta1 must be in [0, 1).")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError("beta2 must be in [0, 1).")
    if momentum_gain < 0.0:
        raise ValueError("momentum_gain must be non-negative.")
    if not (0.0 <= reject_momentum_decay <= 1.0):
        raise ValueError("reject_momentum_decay must be in [0, 1].")
    if grow_factor < 1.0:
        raise ValueError("grow_factor must be >= 1.0.")
    if not (0.0 < shrink_factor <= 1.0):
        raise ValueError("shrink_factor must be in (0, 1].")

    gen = rng if rng is not None else np.random.default_rng(seed)
    tracker = TrackedObjective(objective_fn)
    has_bias_param = _accepts_bias(neighborhood_fn)

    x_curr = _prepare_initial(initial_solution, bounds)
    f_curr = tracker(x_curr)
    x_init = x_curr.copy()
    f_init = f_curr

    sigma = float(initial_step_size)
    upper_sigma = max_step_size
    if upper_sigma is None and bounds is not None:
        upper_sigma = float(np.max(bounds[1] - bounds[0]))

    dim = x_curr.size
    m = np.zeros(dim, dtype=float)
    v = np.zeros(dim, dtype=float)
    t_mom = 0

    history: list[EvaluationRecord] = [
        EvaluationRecord(
            eval_index=tracker.eval_count,
            iter_index=0,
            x=x_curr.copy(),
            f=f_curr,
            accepted=True,
            best_x=x_curr.copy(),
            best_f=f_curr,
            step_size=sigma,
            drift=np.zeros(dim, dtype=float),
            prev_x=x_curr.copy(),
            prev_f=f_curr,
            step_size_before=sigma,
            offset=np.zeros(dim, dtype=float),
            delta=None,
            m_before=np.zeros(dim, dtype=float),
            v_before=np.zeros(dim, dtype=float),
            t_mom_before=0,
            m_hat=np.zeros(dim, dtype=float),
            v_hat=np.zeros(dim, dtype=float),
            d_norm=np.zeros(dim, dtype=float),
            m_after=np.zeros(dim, dtype=float),
            v_after=np.zeros(dim, dtype=float),
            t_mom_after=0,
            accept_streak_before=0,
            reject_streak_before=0,
            accept_streak_after=0,
            reject_streak_after=0,
            step_event="Init",
        )
    ]

    iter_count = 0
    no_improve_count = 0
    accepted_count = 1
    accept_streak = 0
    reject_streak = 0

    while True:
        reason = termination.check(
            eval_count=tracker.eval_count,
            iter_count=iter_count,
            no_improve_count=no_improve_count,
            best_f=f_curr,
            current_step_size=sigma,
            minimize=minimize,
        )
        if reason is not None:
            stop_reason = reason
            break

        iter_count += 1
        prev_x = x_curr.copy()
        prev_f = f_curr
        sigma_before = sigma
        m_before = m.copy()
        v_before = v.copy()
        t_mom_before = t_mom
        acc_before = accept_streak
        rej_before = reject_streak

        if t_mom > 0 and momentum_gain > 0.0:
            m_hat = m / (1.0 - beta1**t_mom) if beta1 > 0.0 else m.copy()
            v_hat = v / (1.0 - beta2**t_mom) if beta2 > 0.0 else v.copy()
            normalized_dir = np.clip(m_hat / (np.sqrt(v_hat) + eps), -1.0, 1.0)
            drift = momentum_gain * sigma * normalized_dir
        else:
            m_hat = np.zeros(dim, dtype=float)
            v_hat = np.zeros(dim, dtype=float)
            normalized_dir = np.zeros(dim, dtype=float)
            drift = np.zeros(dim, dtype=float)

        x_new = _propose_candidate(
            neighborhood_fn=neighborhood_fn,
            has_bias_param=has_bias_param,
            x=prev_x,
            step_size=sigma,
            rng=gen,
            bounds=bounds,
            drift=drift,
        )
        offset = x_new - (prev_x + drift)
        f_new = tracker(x_new)

        if _is_better(f_new, f_curr, minimize=minimize):
            delta = x_new - prev_x
            x_curr = x_new.copy()
            f_curr = f_new
            accepted = True
            accepted_count += 1
            no_improve_count = 0

            t_mom += 1
            m = beta1 * m + (1.0 - beta1) * delta
            v = beta2 * v + (1.0 - beta2) * (delta**2)

            accept_streak += 1
            reject_streak = 0
            event = "Accept"
            if accept_streak >= accept_streak_to_grow:
                sigma *= grow_factor
                if upper_sigma is not None:
                    sigma = min(sigma, upper_sigma)
                accept_streak = 0
                event = "Accept (Grow σ)"
        else:
            delta = None
            accepted = False
            no_improve_count += 1
            accept_streak = 0
            reject_streak += 1
            m = reject_momentum_decay * m
            event = "Reject"
            if reject_streak >= reject_streak_to_shrink:
                sigma = max(min_step_size, sigma * shrink_factor)
                reject_streak = 0
                event = "Reject (Shrink σ)"

        history.append(
            EvaluationRecord(
                eval_index=tracker.eval_count,
                iter_index=iter_count,
                x=x_new.copy(),
                f=f_new,
                accepted=accepted,
                best_x=x_curr.copy(),
                best_f=f_curr,
                step_size=sigma,
                drift=drift.copy(),
                prev_x=prev_x,
                prev_f=prev_f,
                step_size_before=sigma_before,
                offset=offset,
                delta=delta,
                m_before=m_before,
                v_before=v_before,
                t_mom_before=t_mom_before,
                m_hat=m_hat.copy(),
                v_hat=v_hat.copy(),
                d_norm=normalized_dir.copy(),
                m_after=m.copy(),
                v_after=v.copy(),
                t_mom_after=t_mom,
                accept_streak_before=acc_before,
                reject_streak_before=rej_before,
                accept_streak_after=accept_streak,
                reject_streak_after=reject_streak,
                step_event=event,
            )
        )

    return OptimizationResult(
        algorithm_name="Adaptive Momentum Hill Climbing",
        initial_x=x_init,
        initial_f=f_init,
        best_x=x_curr.copy(),
        best_f=f_curr,
        eval_count=tracker.eval_count,
        iter_count=iter_count,
        accepted_count=accepted_count,
        stop_reason=stop_reason,
        history=history,
        params={
            "initial_step_size": float(initial_step_size),
            "beta1": beta1,
            "beta2": beta2,
            "momentum_gain": momentum_gain,
            "reject_momentum_decay": reject_momentum_decay,
            "grow_factor": grow_factor,
            "shrink_factor": shrink_factor,
            "accept_streak_to_grow": accept_streak_to_grow,
            "reject_streak_to_shrink": reject_streak_to_shrink,
            "min_step_size": min_step_size,
            "max_step_size": upper_sigma,
            "eps": eps,
            "minimize": minimize,
            "seed": seed,
            "max_evals": termination.max_evals,
            "max_no_improve": termination.max_no_improve,
        },
    )
