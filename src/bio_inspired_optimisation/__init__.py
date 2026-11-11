"""Core package for Bio-Inspired Optimisation exercises."""

from .benchmarks import BENCHMARKS, BenchmarkSpec, ackley, quadratic, sinusoidal
from .hill_climbing import (
    EvaluationRecord,
    OptimizationResult,
    TrackedObjective,
    adaptive_momentum_hill_climbing,
    simple_hill_climbing,
)
from .neighborhoods import (
    NEIGHBORHOOD_FACTORIES,
    apply_bounds,
    make_gaussian_neighborhood,
    make_uniform_ball_neighborhood,
    make_uniform_box_neighborhood,
)
from .termination import TerminationCriterion

__all__ = [
    "BENCHMARKS",
    "BenchmarkSpec",
    "EvaluationRecord",
    "NEIGHBORHOOD_FACTORIES",
    "OptimizationResult",
    "TerminationCriterion",
    "TrackedObjective",
    "ackley",
    "adaptive_momentum_hill_climbing",
    "apply_bounds",
    "make_gaussian_neighborhood",
    "make_uniform_ball_neighborhood",
    "make_uniform_box_neighborhood",
    "quadratic",
    "simple_hill_climbing",
    "sinusoidal",
]


def main() -> None:
    """CLI entry hint for launching the Streamlit multipage documentation."""
    print("Run the Streamlit app with: uv run streamlit run Home.py")
