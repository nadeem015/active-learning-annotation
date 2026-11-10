"""Uncertainty-based query strategies and a TF-IDF + logistic regression learner."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

STRATEGIES = ("random", "least_confidence", "margin", "entropy")


def uncertainty_scores(probs, strategy, rng=None):
    """Higher score = more informative to label next.

    least_confidence: 1 - max_c p(c|x)
    margin:           1 - (p(top1|x) - p(top2|x))
    entropy:          -sum_c p(c|x) log p(c|x)
    random:           uniform noise (the baseline)
    """
    p = np.asarray(probs, dtype=float)
    if strategy == "random":
        rng = rng if rng is not None else np.random.default_rng()
        return rng.random(len(p))
    if strategy == "least_confidence":
        return 1.0 - p.max(axis=1)
    if strategy == "margin":
        if p.shape[1] < 2:
            return np.zeros(len(p))
        top2 = np.sort(p, axis=1)[:, ::-1][:, :2]
        return 1.0 - (top2[:, 0] - top2[:, 1])
    if strategy == "entropy":
        safe = np.clip(p, 1e-12, 1.0)
        return -(safe * np.log(safe)).sum(axis=1)
    raise ValueError(f"Unknown strategy '{strategy}'. Choose one of {STRATEGIES}.")


class ActiveLearner:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = None

    @property
    def classes(self):
        return [] if self.model is None else [str(c) for c in self.model.classes_]

    def train(self, texts, labels):
        if len(set(labels)) < 2:
            raise ValueError("At least two different labels are needed to train")
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=self.seed)),
        ])
        self.model.fit(texts, labels)

    def predict_proba(self, texts):
        return self.model.predict_proba(texts)

    def predict(self, texts):
        return self.model.predict(texts)

    # kept for backwards compatibility with earlier code
    scores = staticmethod(uncertainty_scores)

    def select(self, texts, batch_size, strategy):
        """Return (indices, scores, probabilities) of the batch to annotate next."""
        probs = self.predict_proba(texts)
        s = uncertainty_scores(probs, strategy, self.rng)
        batch_size = min(batch_size, len(texts))
        idx = np.argsort(-s, kind="stable")[:batch_size]
        return idx.tolist(), s[idx].tolist(), probs[idx].tolist()

    def evaluate(self, texts, labels):
        pred = self.predict(texts)
        pr, re, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
        return {"accuracy": float(accuracy_score(labels, pred)), "precision_macro": float(pr),
                "recall_macro": float(re), "f1_macro": float(f1)}
