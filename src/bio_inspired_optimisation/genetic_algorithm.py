"""Exercise 2: reproducible genetic algorithm for grayscale image reconstruction.

Population chromosomes are uint8 arrays with one gene per pixel (values 0..255).
Fitness is normalized pixel-wise mean squared error, so it is comparable across
image sizes. Optimization minimizes this loss; no gradients are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

Initialization = Literal["random", "mean_tone", "hybrid"]
ParentSelection = Literal["tournament", "rank"]
Crossover = Literal["uniform", "two_point", "rectangle"]
Mutation = Literal["gaussian", "reset"]
SurvivorSelection = Literal["elitism", "plus"]


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 48
    generations: int = 120
    initialization: Initialization = "hybrid"
    parent_selection: ParentSelection = "tournament"
    tournament_size: int = 3
    crossover: Crossover = "uniform"
    crossover_rate: float = 0.90
    mutation: Mutation = "gaussian"
    mutation_rate: float = 0.06
    mutation_strength: float = 35.0
    survivor_selection: SurvivorSelection = "elitism"
    elites: int = 2
    target_mse: float | None = None
    seed: int = 123


@dataclass(frozen=True)
class GARecord:
    generation: int
    eval_count: int
    best_mse: float
    mean_mse: float
    diversity: float


@dataclass(frozen=True)
class GAResult:
    best_image: np.ndarray
    best_mse: float
    initial_best_mse: float
    eval_count: int
    runtime_seconds: float
    stop_reason: str
    config: GAConfig
    history: tuple[GARecord, ...]
    best_frames: tuple[np.ndarray, ...]


def normalized_mse(population: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Evaluate every individual exactly once; result range [0, 1].

    population shape: (population_size, number_of_pixels).
    target shape: (height, width) or (number_of_pixels,).
    """
    pop = np.asarray(population)
    goal = np.asarray(target)
    if pop.ndim != 2 or goal.size != pop.shape[1]:
        raise ValueError("Population and target must have matching pixel counts")
    diff = (pop.astype(np.float32) - goal.reshape(1, -1).astype(np.float32)) / 255.0
    return np.mean(diff * diff, axis=1, dtype=np.float64)


def _validate(target: np.ndarray, c: GAConfig) -> np.ndarray:
    t = np.asarray(target)
    if t.ndim != 2 or min(t.shape) < 2:
        raise ValueError("Target image must be a 2D grayscale image of at least 2x2 pixels")
    if t.dtype != np.uint8:
        raise ValueError("Target image must use uint8 pixel values from 0 to 255")
    if c.population_size < 4 or c.generations < 0:
        raise ValueError("Population size must be >= 4 and generations must be >= 0")
    if c.elites < 1 or c.elites >= c.population_size:
        raise ValueError("Elites must be between 1 and population_size - 1")
    if not 2 <= c.tournament_size <= c.population_size:
        raise ValueError("Tournament size must be in [2, population_size]")
    if not 0 <= c.crossover_rate <= 1 or not 0 <= c.mutation_rate <= 1:
        raise ValueError("Crossover rate and mutation rate must be in [0, 1]")
    if not np.isfinite(c.mutation_strength) or c.mutation_strength < 0:
        raise ValueError("Mutation strength must be nonnegative and finite")
    if c.target_mse is not None and not 0 <= c.target_mse <= 1:
        raise ValueError("Target MSE must be in [0, 1]")
    if c.initialization not in ("random", "mean_tone", "hybrid"):
        raise ValueError("Unknown initialization strategy")
    if c.parent_selection not in ("tournament", "rank"):
        raise ValueError("Unknown parent selection strategy")
    if c.crossover not in ("uniform", "two_point", "rectangle"):
        raise ValueError("Unknown crossover strategy")
    if c.mutation not in ("gaussian", "reset"):
        raise ValueError("Unknown mutation strategy")
    if c.survivor_selection not in ("elitism", "plus"):
        raise ValueError("Unknown survivor selection strategy")
    return t


def _initialize(target: np.ndarray, c: GAConfig, rng: np.random.Generator) -> np.ndarray:
    count, n = c.population_size, target.size
    if c.initialization == "random":
        return rng.integers(0, 256, (count, n), dtype=np.uint8)
    # Uses ONLY the global average brightness, not any target pixel positions.
    mean = float(np.mean(target))
    tone_population = np.clip(
        np.rint(rng.normal(mean, 55.0, size=(count, n))), 0, 255
    ).astype(np.uint8)
    if c.initialization == "mean_tone":
        return tone_population
    # Hybrid: some tone-aware individuals and some fully random ones.
    random_count = count // 2
    tone_population[:random_count] = rng.integers(
        0, 256, (random_count, n), dtype=np.uint8
    )
    return tone_population


def _choose_parent(fitness: np.ndarray, c: GAConfig, rng: np.random.Generator) -> int:
    n = len(fitness)
    if c.parent_selection == "tournament":
        contenders = rng.choice(n, size=c.tournament_size, replace=False)
        return int(contenders[np.argmin(fitness[contenders])])
    # Linear ranking is well-defined even if several individuals have MSE == 0.
    order = np.argsort(fitness, kind="stable")
    weights = np.empty(n, dtype=np.float64)
    weights[order] = np.arange(n, 0, -1, dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(n, p=weights))


def _crossover(
    left: np.ndarray,
    right: np.ndarray,
    shape: tuple[int, int],
    c: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if rng.random() >= c.crossover_rate:
        return (left if rng.random() < 0.5 else right).copy()
    n = left.size
    if c.crossover == "uniform":
        from_left = rng.random(n) < 0.5
        return np.where(from_left, left, right).astype(np.uint8)
    if c.crossover == "two_point":
        cuts = np.sort(rng.choice(np.arange(1, n), size=2, replace=False))
        child = left.copy()
        child[cuts[0]:cuts[1]] = right[cuts[0]:cuts[1]]
        return child
    # Spatial rectangle crossover: preserve a coherent image patch from parent B.
    h, w = shape
    top = int(rng.integers(0, h))
    bottom = int(rng.integers(top + 1, h + 1))
    x0 = int(rng.integers(0, w))
    x1 = int(rng.integers(x0 + 1, w + 1))
    child = left.reshape(shape).copy()
    child[top:bottom, x0:x1] = right.reshape(shape)[top:bottom, x0:x1]
    return child.reshape(-1)


def _mutate(child: np.ndarray, c: GAConfig, rng: np.random.Generator) -> None:
    mask = rng.random(child.size) < c.mutation_rate
    count = int(mask.sum())
    if count == 0:
        return
    if c.mutation == "reset":
        child[mask] = rng.integers(0, 256, size=count, dtype=np.uint8)
    else:
        values = child[mask].astype(np.float32) + rng.normal(0.0, c.mutation_strength, count)
        child[mask] = np.clip(np.rint(values), 0, 255).astype(np.uint8)


def run_genetic_algorithm(target: np.ndarray, config: GAConfig = GAConfig()) -> GAResult:
    """Optimize a grayscale target; record every generation's best image and MSE.

    Evaluation count is exactly the number of objective function calculations:
    population_size at generation 0 plus all newly evaluated offspring.
    Parents / retained elites are NOT re-evaluated.
    """
    t = _validate(target, config)
    start = perf_counter()
    rng = np.random.default_rng(config.seed)
    pop = _initialize(t, config, rng)
    fitness = normalized_mse(pop, t)
    count = config.population_size

    history: list[GARecord] = []
    best_frames: list[np.ndarray] = []
    best_ever = float("inf")
    best_chromosome: np.ndarray | None = None

    def record(generation: int) -> None:
        nonlocal best_ever, best_chromosome
        index = int(np.argmin(fitness))
        val = float(fitness[index])
        if val < best_ever:
            best_ever = val
            best_chromosome = pop[index].copy()
        assert best_chromosome is not None
        diversity = float(np.mean(np.std(pop.astype(np.float32), axis=0)) / 255.0)
        history.append(
            GARecord(
                generation=generation,
                eval_count=count,
                best_mse=best_ever,
                mean_mse=float(np.mean(fitness)),
                diversity=diversity,
            )
        )
        best_frames.append(best_chromosome.reshape(t.shape).copy())

    record(0)
    initial_best = best_ever
    stop_reason = "Maximum generations reached"
    if config.target_mse is not None and best_ever <= config.target_mse:
        stop_reason = "Target MSE reached"
    else:
        for generation in range(1, config.generations + 1):
            if config.survivor_selection == "elitism":
                elite_idx = np.argsort(fitness)[:config.elites]
                keep = pop[elite_idx].copy()
                keep_fitness = fitness[elite_idx].copy()
                num_offspring = config.population_size - config.elites
            else:
                num_offspring = config.population_size

            offspring = np.empty((num_offspring, t.size), dtype=np.uint8)
            for i in range(num_offspring):
                pa = pop[_choose_parent(fitness, config, rng)]
                pb = pop[_choose_parent(fitness, config, rng)]
                child = _crossover(pa, pb, t.shape, config, rng)
                _mutate(child, config, rng)
                offspring[i] = child

            offspring_fitness = normalized_mse(offspring, t)
            count += num_offspring
            if config.survivor_selection == "elitism":
                pop = np.concatenate([keep, offspring], axis=0)
                fitness = np.concatenate([keep_fitness, offspring_fitness])
            else:
                pool = np.concatenate([pop, offspring], axis=0)
                pool_fitness = np.concatenate([fitness, offspring_fitness])
                survivors = np.argsort(pool_fitness)[:config.population_size]
                pop, fitness = pool[survivors].copy(), pool_fitness[survivors].copy()

            record(generation)
            if config.target_mse is not None and best_ever <= config.target_mse:
                stop_reason = "Target MSE reached"
                break

    assert best_chromosome is not None
    return GAResult(
        best_image=best_chromosome.reshape(t.shape).copy(),
        best_mse=best_ever,
        initial_best_mse=initial_best,
        eval_count=count,
        runtime_seconds=perf_counter() - start,
        stop_reason=stop_reason,
        config=config,
        history=tuple(history),
        best_frames=tuple(best_frames),
    )
