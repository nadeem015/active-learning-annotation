"""Generate a synthetic customer-feedback dataset for the demo and experiments.

The sentences are built from templates, so they are NOT real data. They are
varied enough (subjects x phrasings x hedges, plus 'X but Y' sentences whose
label follows the clause after 'but') that the model has to learn something
and that uncertainty sampling has informative examples to find.

    python scripts/generate_demo_data.py --rows 600 --seed 42
"""

import argparse
import csv
import os
import random

SUBJECTS = ["the support team", "the delivery", "the mobile app", "the refund process", "the staff",
            "the website", "the checkout", "the product", "customer service", "the installation",
            "the billing team", "the new update", "the onboarding", "the return policy", "the chat agent"]

PHRASES = {
    "positive": ["was fantastic", "solved my problem quickly", "exceeded my expectations", "was really helpful",
                 "made everything easy", "works perfectly now", "was friendly and professional",
                 "was faster than I expected", "deserves five stars", "made me a happy customer",
                 "was smooth from start to finish", "impressed me"],
    "negative": ["was a complete disaster", "never answered my emails", "wasted hours of my time",
                 "was rude and unhelpful", "keeps crashing", "charged me twice", "was painfully slow",
                 "broke after one day", "made me want to cancel", "was confusing and frustrating",
                 "ignored my complaint", "let me down badly"],
    "neutral": ["was updated on Monday", "is available on weekdays", "was used for my order",
                "has a new layout", "sent a confirmation email", "requires an account", "was contacted twice",
                "is located in the settings menu", "changed its opening hours", "asked for my order number",
                "uses a reference code", "was mentioned in the newsletter"],
}
HEDGES = ["", "", "", "honestly ", "overall ", "I think ", "to be fair ", "as far as I can tell "]
ENDINGS = ["", "", ".", "!", " this week.", " yesterday.", " again."]


def sentence(rng, label):
    return f"{rng.choice(HEDGES)}{rng.choice(SUBJECTS)} {rng.choice(PHRASES[label])}{rng.choice(ENDINGS)}".strip()


def make_rows(n, seed):
    rng = random.Random(seed)
    labels = list(PHRASES)
    rows = []
    for i in range(n):
        label = labels[i % 3]
        if rng.random() < 0.25:
            # Mixed sentence: the clause after "but" decides the label.
            other = rng.choice([l for l in labels if l != label])
            text = f"{sentence(rng, other).rstrip('.!')} but {sentence(rng, label)}"
        else:
            text = sentence(rng, label)
        rows.append({"text": text[0].upper() + text[1:], "label": label})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "demo_pool.csv"))
    a = p.parse_args()
    rows = make_rows(a.rows, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)
    print("Generated", len(rows), "rows ->", os.path.normpath(a.out))
