"""Termination criteria for iterative optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminationCriterion:
    """Reusable stopping conditions for Hill Climbing and future optimizers."""

    max_evals: int = 400
    max_iters: int | None = None
    max_no_improve: int | None = None
    min_step_size: float | None = None
    target_fitness: float | None = None

    def __post_init__(self) -> None:
        if self.max_evals < 1:
            raise ValueError("max_evals must be at least 1.")
        if self.max_iters is not None and self.max_iters < 0:
            raise ValueError("max_iters must be non-negative.")
        if self.max_no_improve is not None and self.max_no_improve < 1:
            raise ValueError("max_no_improve must be at least 1.")
        if self.min_step_size is not None and self.min_step_size <= 0.0:
            raise ValueError("min_step_size must be positive.")

    def check(
        self,
        *,
        eval_count: int,
        iter_count: int,
        no_improve_count: int,
        best_f: float,
        current_step_size: float,
        minimize: bool = True,
    ) -> str | None:
        """Return a reason string if termination is triggered, else None."""
        if self.target_fitness is not None:
            reached = (best_f <= self.target_fitness) if minimize else (best_f >= self.target_fitness)
            if reached:
                return "target_fitness"
        if eval_count >= self.max_evals:
            return "max_evals"
        if self.max_iters is not None and iter_count >= self.max_iters:
            return "max_iters"
        if self.max_no_improve is not None and no_improve_count >= self.max_no_improve:
            return "max_no_improve"
        if self.min_step_size is not None and current_step_size <= self.min_step_size:
            return "min_step_size"
        return None
