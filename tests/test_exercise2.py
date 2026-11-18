"""Tests for Exercise 2 GA fitness, operators, evaluation counts, and replay."""

from dataclasses import replace

import numpy as np
import pytest

from bio_inspired_optimisation.genetic_algorithm import GAConfig, normalized_mse, run_genetic_algorithm


@pytest.fixture
def target() -> np.ndarray:
    return np.arange(256, dtype=np.uint8).reshape(16, 16)


def test_fitness_is_normalized_and_vectorized(target: np.ndarray) -> None:
    same = target.reshape(1, -1)
    white = np.full((1, target.size), 255, dtype=np.uint8)
    assert normalized_mse(same, target)[0] == 0.0
    assert normalized_mse(np.zeros((1, 4), dtype=np.uint8), np.full((2, 2), 255, np.uint8))[0] == pytest.approx(1.0)
    assert 0 <= normalized_mse(white, target)[0] <= 1
    with pytest.raises(ValueError):
        normalized_mse(np.zeros((4, 10), dtype=np.uint8), target)


def test_replay_and_exact_eval_accounting_elitism(target: np.ndarray) -> None:
    config = GAConfig(population_size=20, generations=12, elites=2, seed=123)
    result = run_genetic_algorithm(target, config)
    assert len(result.history) == len(result.best_frames) == 13
    assert result.eval_count == 20 + 12 * (20 - 2)
    assert [r.eval_count for r in result.history] == [20 + i * 18 for i in range(13)]
    assert np.all(np.diff([r.best_mse for r in result.history]) <= 1e-15)
    assert result.best_mse <= result.initial_best_mse
    assert result.best_image.shape == target.shape
    assert result.best_image.dtype == np.uint8
    assert np.array_equal(result.best_frames[-1], result.best_image)
    assert all(0 <= r.diversity <= 0.5 for r in result.history)


def test_plus_selection_evaluates_all_offspring(target: np.ndarray) -> None:
    config = GAConfig(population_size=16, generations=5, survivor_selection="plus")
    result = run_genetic_algorithm(target, config)
    assert result.eval_count == 16 + 5 * 16
    assert len(result.history) == 6
    assert np.all(np.diff([r.best_mse for r in result.history]) <= 0)


def test_reproducible_excluding_wall_clock(target: np.ndarray) -> None:
    c = GAConfig(population_size=16, generations=10, seed=42)
    a, b = run_genetic_algorithm(target, c), run_genetic_algorithm(target, c)
    assert a.best_mse == b.best_mse
    assert np.array_equal(a.best_image, b.best_image)
    assert a.history == b.history
    assert all(np.array_equal(x, y) for x, y in zip(a.best_frames, b.best_frames))


def test_target_termination_generation_zero(target: np.ndarray) -> None:
    c = GAConfig(population_size=12, generations=100, target_mse=1.0)
    result = run_genetic_algorithm(target, c)
    assert len(result.history) == 1
    assert result.eval_count == 12
    assert result.stop_reason == "Target MSE reached"


def test_all_operator_choices_execute(target: np.ndarray) -> None:
    base = GAConfig(population_size=10, generations=2, elites=1, mutation_rate=0.12)
    variants = [
        replace(base, initialization="random"),
        replace(base, initialization="mean_tone"),
        replace(base, parent_selection="rank"),
        replace(base, crossover="rectangle"),
        replace(base, crossover="two_point"),
        replace(base, mutation="reset"),
    ]
    for c in variants:
        result = run_genetic_algorithm(target, c)
        assert result.eval_count == 28
        assert len(result.best_frames) == 3


def test_config_validation(target: np.ndarray) -> None:
    with pytest.raises(ValueError):
        run_genetic_algorithm(target.astype(np.int16))
    with pytest.raises(ValueError):
        run_genetic_algorithm(target, GAConfig(population_size=3))
    with pytest.raises(ValueError):
        run_genetic_algorithm(target, GAConfig(mutation_rate=1.5))
