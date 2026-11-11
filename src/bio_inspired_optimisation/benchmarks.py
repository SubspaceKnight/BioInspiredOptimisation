"""Benchmark objective functions for Bio-Inspired Optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


def _to_1d_array(x: ArrayLike | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector or scalar, got shape {arr.shape}.")
    return arr


def quadratic(x: ArrayLike | float) -> float:
    """Quadratic benchmark: f(x) = x^2 in 1D, or sum(x_i^2) in nD."""
    vec = _to_1d_array(x)
    return float(np.sum(vec**2))


def sinusoidal(x: ArrayLike | float) -> float:
    """Sinusoidal benchmark: f(x) = sin(x) in 1D, or sum(sin(x_i)) in nD."""
    vec = _to_1d_array(x)
    if vec.size == 1:
        return float(np.sin(vec[0]))
    return float(np.sum(np.sin(vec)))


def ackley(
    x: ArrayLike | float,
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2.0 * np.pi,
) -> float:
    """Ackley benchmark function in n dimensions (default a=20, b=0.2, c=2*pi)."""
    vec = _to_1d_array(x)
    n = vec.size
    sum_sq = np.mean(vec**2)
    sum_cos = np.mean(np.cos(c * vec))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq))
    term2 = -np.exp(sum_cos)
    return float(term1 + term2 + a + np.e)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata and helpers for a benchmark optimization function."""

    key: str
    name: str
    fn: Callable[[np.ndarray], float]
    default_dim: int
    default_bounds: tuple[float, float]
    global_min_f: float
    global_min_description: str
    formula_latex: str
    landscape_notes: str

    def global_min_x(self, dim: int | None = None) -> np.ndarray:
        """Return a canonical global minimizer in the requested dimension."""
        d = self.default_dim if dim is None else dim
        if self.key == "sinusoidal":
            return np.full(d, -0.5 * np.pi, dtype=float)
        return np.zeros(d, dtype=float)

    def make_bounds(
        self,
        dim: int | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct (lower, upper) bound arrays of length `dim`."""
        d = self.default_dim if dim is None else dim
        lo = self.default_bounds[0] if low is None else float(low)
        hi = self.default_bounds[1] if high is None else float(high)
        if lo >= hi:
            raise ValueError(f"Lower bound ({lo}) must be strictly less than upper bound ({hi}).")
        return np.full(d, lo, dtype=float), np.full(d, hi, dtype=float)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "quadratic": BenchmarkSpec(
        key="quadratic",
        name="Quadratic (1D)",
        fn=quadratic,
        default_dim=1,
        default_bounds=(-10.0, 10.0),
        global_min_f=0.0,
        global_min_description="x* = 0, f(x*) = 0",
        formula_latex=r"f(x) = x^2",
        landscape_notes="Strictly convex and unimodal; tests convergence speed and final step-size precision.",
    ),
    "sinusoidal": BenchmarkSpec(
        key="sinusoidal",
        name="Sinusoidal (1D)",
        fn=sinusoidal,
        default_dim=1,
        default_bounds=(-2.0 * np.pi, 2.0 * np.pi),
        global_min_f=-1.0,
        global_min_description=r"x* = -π/2 + 2πk, f(x*) = -1",
        formula_latex=r"f(x) = \sin(x)",
        landscape_notes="Periodic and multimodal on bounded intervals; demonstrates basin selection and local/boundary traps.",
    ),
    "ackley": BenchmarkSpec(
        key="ackley",
        name="Ackley (2D)",
        fn=ackley,
        default_dim=2,
        default_bounds=(-10.0 * np.pi, 10.0 * np.pi),
        global_min_f=0.0,
        global_min_description="x* = (0, 0), f(x*) = 0",
        formula_latex=(
            r"f(\vec{x}) = -a\exp\!\left(-b\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}\right)"
            r" - \exp\!\left(\frac{1}{n}\sum_{i=1}^{n}\cos(c x_i)\right) + a + e"
        ),
        landscape_notes=(
            "Flat outer plateau transitioning into a deep central funnel surrounded by many "
            "cosine-induced local minima (a=20, b=0.2, c=2π)."
        ),
    ),
}


def evaluate_grid_1d(
    fn: Callable[[np.ndarray], float],
    low: float,
    high: float,
    num_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 1D benchmark function over a regular grid."""
    xs = np.linspace(low, high, num_points)
    ys = np.array([fn(np.array([x])) for x in xs], dtype=float)
    return xs, ys


def evaluate_grid_2d(
    fn: Callable[[np.ndarray], float],
    low_x: float,
    high_x: float,
    low_y: float,
    high_y: float,
    num_points: int = 140,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a 2D benchmark function over a regular meshgrid."""
    xs = np.linspace(low_x, high_x, num_points)
    ys = np.linspace(low_y, high_y, num_points)
    gx, gy = np.meshgrid(xs, ys)
    if fn is ackley:
        r2 = 0.5 * (gx**2 + gy**2)
        cos_mean = 0.5 * (np.cos(2.0 * np.pi * gx) + np.cos(2.0 * np.pi * gy))
        gz = -20.0 * np.exp(-0.2 * np.sqrt(r2)) - np.exp(cos_mean) + 20.0 + np.e
        return xs, ys, gz
    gz = np.empty_like(gx, dtype=float)
    for i in range(num_points):
        for j in range(num_points):
            gz[i, j] = fn(np.array([gx[i, j], gy[i, j]]))
    return xs, ys, gz
