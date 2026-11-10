"""Plot mean learning curves (+/- 1 std over seeds) for each query strategy."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {"random": ("#7f7f7f", "--", "o"), "least_confidence": ("#1f77b4", "-", "s"),
         "margin": ("#2ca02c", "-", "^"), "entropy": ("#d62728", "-", "D")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="research/results.csv")
    p.add_argument("--output", default="research/figures")
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)
    d = pd.read_csv(a.input)
    n_seeds = d["seed"].nunique() if "seed" in d else 1
    for metric, name in [("f1_macro", "Macro F1"), ("accuracy", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s, g in d.groupby("strategy"):
            m = g.groupby("labeled_samples")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            colour, ls, marker = STYLE.get(s, (None, "-", "o"))
            ax.plot(m.labeled_samples, m["mean"], ls, marker=marker, ms=4, color=colour, label=s.replace("_", " "))
            ax.fill_between(m.labeled_samples, m["mean"] - m["std"], m["mean"] + m["std"], color=colour, alpha=0.12)
        ax.set_xlabel("Labelled examples (annotation budget)")
        ax.set_ylabel(f"Test {name}")
        ax.set_title(f"{name} vs. annotation budget (mean ± std, {n_seeds} seeds)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.output, f"{metric}.{ext}"), dpi=200)
        plt.close(fig)
    print("Saved figures to", a.output)


if __name__ == "__main__":
    main()
