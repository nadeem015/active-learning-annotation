import os

from research.run_experiment import run, summarise

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "demo_pool.csv")


def test_experiment_runs_all_strategies_with_equal_budgets():
    rows = run(DATA, seeds=1, cycles=2, seed_size=9, batch=5)
    assert {r["strategy"] for r in rows} == {"random", "least_confidence", "margin", "entropy"}
    budgets = {r["strategy"]: [x["labeled_samples"] for x in rows if x["strategy"] == r["strategy"]] for r in rows}
    assert all(b == [9, 14, 19] for b in budgets.values())
    curve, alc = summarise(rows)
    assert len(alc) == 4
