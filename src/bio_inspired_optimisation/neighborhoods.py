"""Continuous-domain neighborhood generators for Hill Climbing."""

from __future__ import annotations

from typing import Callable, Literal, Protocol

import numpy as np

Bounds = tuple[np.ndarray, np.ndarray]
BoundaryMode = Literal["clip", "reflect"]


class NeighborhoodFn(Protocol):
    """Protocol for continuous neighborhood functions."""

    def __call__(
        self,
        x: np.ndarray,
        step_size: float | np.ndarray,
        rng: np.random.Generator,
        bounds: Bounds | None = None,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


def apply_bounds(
    x: np.ndarray,
    bounds: Bounds | None,
    mode: BoundaryMode = "clip",
) -> np.ndarray:
    """Enforce domain bounds using clipping or mirror reflection."""
    if bounds is None:
        return x
    low, high = bounds
    if mode == "clip":
        return np.clip(x, low, high)
    if mode == "reflect":
        span = high - low
        shifted = x - low
        # Reflect smoothly across both boundaries even for multi-span jumps
        period = 2.0 * span
        mod = np.mod(shifted, period)
        reflected = np.where(mod <= span, mod, period - mod)
        return low + reflected
    raise ValueError(f"Unsupported boundary mode: {mode}")


def make_uniform_box_neighborhood(
    boundary_mode: BoundaryMode = "clip",
) -> NeighborhoodFn:
    """Sample delta uniformly in [-step_size, +step_size]^n around x + bias."""

    def sample(
        x: np.ndarray,
        step_size: float | np.ndarray,
        rng: np.random.Generator,
        bounds: Bounds | None = None,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        step = np.asarray(step_size, dtype=float)
        center = x if bias is None else (x + bias)
        delta = rng.uniform(-step, step, size=x.shape)
        return apply_bounds(center + delta, bounds, mode=boundary_mode)

    return sample


def make_uniform_ball_neighborhood(
    boundary_mode: BoundaryMode = "clip",
) -> NeighborhoodFn:
    """Sample delta uniformly inside the Euclidean ball ||delta||_2 <= step_size."""

    def sample(
        x: np.ndarray,
        step_size: float | np.ndarray,
        rng: np.random.Generator,
        bounds: Bounds | None = None,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        center = x if bias is None else (x + bias)
        dim = x.size
        direction = rng.normal(0.0, 1.0, size=x.shape)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            unit = np.zeros_like(x)
        else:
            unit = direction / norm
        radius_scale = float(rng.random() ** (1.0 / dim))
        delta = unit * radius_scale * np.asarray(step_size, dtype=float)
        return apply_bounds(center + delta, bounds, mode=boundary_mode)

    return sample


def make_gaussian_neighborhood(
    boundary_mode: BoundaryMode = "clip",
) -> NeighborhoodFn:
    """Sample delta ~ Normal(0, step_size^2 * I) around x + bias."""

    def sample(
        x: np.ndarray,
        step_size: float | np.ndarray,
        rng: np.random.Generator,
        bounds: Bounds | None = None,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        center = x if bias is None else (x + bias)
        delta = rng.normal(0.0, np.asarray(step_size, dtype=float), size=x.shape)
        return apply_bounds(center + delta, bounds, mode=boundary_mode)

    return sample


NEIGHBORHOOD_FACTORIES: dict[str, Callable[[BoundaryMode], NeighborhoodFn]] = {
    "Uniform Box [-δ, δ]ⁿ": make_uniform_box_neighborhood,
    "Uniform L2 Ball (||Δx||₂ ≤ δ)": make_uniform_ball_neighborhood,
    "Gaussian N(0, δ²I)": make_gaussian_neighborhood,
}
