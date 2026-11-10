import numpy as np
import pytest

from app.active_learning import STRATEGIES, ActiveLearner, uncertainty_scores

CONFIDENT, UNSURE = [0.9, 0.05, 0.05], [0.4, 0.35, 0.25]


@pytest.mark.parametrize("strategy", ["least_confidence", "margin", "entropy"])
def test_unsure_examples_score_higher(strategy):
    s = uncertainty_scores(np.array([CONFIDENT, UNSURE]), strategy)
    assert s[1] > s[0]


def test_known_values():
    p = np.array([[0.5, 0.5]])
    assert uncertainty_scores(p, "least_confidence")[0] == pytest.approx(0.5)
    assert uncertainty_scores(p, "margin")[0] == pytest.approx(1.0)
    assert uncertainty_scores(p, "entropy")[0] == pytest.approx(np.log(2))


def test_entropy_handles_zero_probabilities():
    s = uncertainty_scores(np.array([[1.0, 0.0, 0.0]]), "entropy")
    assert np.isfinite(s).all() and s[0] == pytest.approx(0, abs=1e-9)


def test_random_is_reproducible_with_seed():
    p = np.full((10, 2), 0.5)
    a = uncertainty_scores(p, "random", np.random.default_rng(1))
    b = uncertainty_scores(p, "random", np.random.default_rng(1))
    assert np.allclose(a, b)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        uncertainty_scores(np.array([CONFIDENT]), "magic")


TEXTS = ["great service", "terrible service", "fantastic staff", "awful staff",
         "the office opens at nine", "the form has two pages"]
LABELS = ["positive", "negative", "positive", "negative", "neutral", "neutral"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_select_returns_most_uncertain_batch(strategy):
    learner = ActiveLearner(seed=0)
    learner.train(TEXTS, LABELS)
    pool = ["great staff", "service", "terrible form", "nine pages", "awful"]
    idx, scores, probs = learner.select(pool, 3, strategy)
    assert len(idx) == 3 and len(set(idx)) == 3
    assert all(0 <= i < len(pool) for i in idx)
    assert scores == sorted(scores, reverse=True)
    assert np.allclose(np.sum(probs, axis=1), 1)


def test_training_needs_two_classes():
    with pytest.raises(ValueError):
        ActiveLearner().train(["a", "b"], ["x", "x"])


def test_evaluate_perfect_on_training_data():
    learner = ActiveLearner()
    learner.train(TEXTS * 3, LABELS * 3)
    m = learner.evaluate(TEXTS, LABELS)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0
