"""FastAPI service for an uncertainty-driven active-learning annotation loop.

Cycle: train on labelled data -> score the unlabelled pool -> a human labels the
most uncertain batch -> retrain -> evaluate on a held-out test set -> repeat.
"""

import csv
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .active_learning import ActiveLearner
from .config import settings
from .db import SessionLocal, init_db
from .models import AnnotationEvent, Document, Experiment, Prediction
from .schemas import AnnotationBatch, InitRequest, QueryRequest

ROOT = Path(__file__).resolve().parent.parent
UI_FILE = Path(__file__).resolve().parent / "ui.html"
learner = ActiveLearner(settings.random_seed)


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title="Uncertainty-Driven Active Learning", version="1.1.0", lifespan=lifespan)


def data_file() -> Path:
    p = Path(settings.data_path)
    return p if p.is_absolute() else ROOT / p


def get_experiment(db) -> Experiment:
    exp = db.query(Experiment).first()
    if exp is None:
        raise HTTPException(400, "No experiment yet. Call POST /api/v1/initialize first.")
    return exp


def train_model(db):
    rows = db.query(Document).filter(Document.labeled.is_(True), Document.split == "train").all()
    labels = [r.label for r in rows]
    if len(set(labels)) < 2:
        raise HTTPException(400, "At least two classes are required to train the model")
    learner.train([r.text for r in rows], labels)
    return len(rows)


def evaluation(db):
    rows = db.query(Document).filter(Document.split == "test").all()
    return learner.evaluate([r.text for r in rows], [r.label for r in rows]) if rows else None


@app.get("/health")
def health():
    return {"status": "ok", "model_backend": settings.model_backend}


@app.get("/api/v1/status")
def status():
    with SessionLocal() as db:
        exp = db.query(Experiment).first()
        return {
            "experiment": exp.id if exp else None,
            "cycle": exp.current_cycle if exp else 0,
            "strategy": exp.strategy if exp else None,
            "model_version": exp.model_version if exp else None,
            "labeled": db.query(Document).filter(Document.labeled.is_(True), Document.split == "train").count(),
            "unlabeled_pool": db.query(Document).filter(Document.labeled.is_(False), Document.split == "pool").count(),
            "test_set": db.query(Document).filter(Document.split == "test").count(),
        }


@app.get("/api/v1/labels")
def labels():
    """Label set the annotator can choose from (taken from the data file)."""
    path = data_file()
    if not path.exists():
        return {"labels": []}
    with open(path, encoding="utf-8") as f:
        return {"labels": sorted({r["label"] for r in csv.DictReader(f) if r.get("label")})}


@app.post("/api/v1/initialize")
def initialize(req: InitRequest):
    path = data_file()
    if not path.exists():
        raise HTTPException(400, f"Data file not found: {path}. Run scripts/generate_demo_data.py first.")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < req.seed_size + settings.test_size + 1:
        raise HTTPException(400, "Data file is too small for this seed size")

    with SessionLocal() as db:
        for model in (Prediction, AnnotationEvent, Experiment, Document):
            db.query(model).delete()
        random.Random(settings.random_seed).shuffle(rows)
        test, remaining = rows[-settings.test_size:], rows[:-settings.test_size]
        seed, pool = remaining[:req.seed_size], remaining[req.seed_size:]
        for r in seed:
            db.add(Document(text=r["text"], label=r["label"], split="train", labeled=True, cycle=0))
        for r in pool:  # the pool's true labels are hidden from the model and the annotator
            db.add(Document(text=r["text"], split="pool", labeled=False, cycle=0))
        for r in test:
            db.add(Document(text=r["text"], label=r["label"], split="test", labeled=True, cycle=0))
        exp = Experiment(strategy=req.strategy, batch_size=req.batch_size, current_cycle=0, model_version="v0")
        db.add(exp)
        db.commit()
        train_model(db)
        return {"status": "initialized", "experiment_id": exp.id, "seed_size": len(seed),
                "pool_size": len(pool), "test_size": len(test), "metrics": evaluation(db)}


@app.post("/api/v1/query")
def query(req: QueryRequest):
    with SessionLocal() as db:
        exp = get_experiment(db)
        train_model(db)
        docs = db.query(Document).filter(Document.labeled.is_(False), Document.split == "pool").all()
        if not docs:
            return {"strategy": req.strategy, "cycle": exp.current_cycle, "selected": []}
        idx, scores, probs = learner.select([d.text for d in docs], req.batch_size, req.strategy)
        classes = learner.classes
        selected = []
        for i, s, p in zip(idx, scores, probs):
            d = docs[i]
            best = max(range(len(p)), key=p.__getitem__)
            db.add(Prediction(document_id=d.id, cycle=exp.current_cycle, strategy=req.strategy,
                              predicted_label=classes[best], confidence=float(p[best]), uncertainty=float(s)))
            selected.append({"id": d.id, "text": d.text, "predicted_label": classes[best],
                             "confidence": round(float(p[best]), 4), "uncertainty": round(float(s), 4)})
        exp.strategy, exp.batch_size = req.strategy, req.batch_size
        db.commit()
        return {"strategy": req.strategy, "cycle": exp.current_cycle, "selected": selected}


@app.post("/api/v1/annotate")
def annotate(batch: AnnotationBatch):
    """Store human labels, move the documents to the training set and retrain."""
    with SessionLocal() as db:
        exp = get_experiment(db)
        accepted = 0
        for a in batch.annotations:
            d = db.get(Document, a.document_id)
            if d is None or d.labeled or d.split != "pool":
                continue
            d.label, d.labeled, d.split, d.cycle = a.label, True, "train", exp.current_cycle + 1
            db.add(AnnotationEvent(document_id=d.id, cycle=exp.current_cycle, label=a.label))
            accepted += 1
        if accepted == 0:
            raise HTTPException(400, "None of the documents could be annotated (unknown or already labelled)")
        exp.current_cycle += 1
        exp.model_version = f"v{exp.current_cycle}"
        db.commit()
        n_train = train_model(db)
        return {"status": "annotated", "accepted": accepted, "cycle": exp.current_cycle,
                "model_version": exp.model_version, "training_size": n_train, "metrics": evaluation(db)}


@app.post("/api/v1/retrain")
def retrain():
    with SessionLocal() as db:
        exp = get_experiment(db)
        n_train = train_model(db)
        return {"status": "trained", "model_version": exp.model_version, "training_size": n_train}


@app.get("/api/v1/evaluate")
def evaluate():
    with SessionLocal() as db:
        get_experiment(db)
        train_model(db)
        return evaluation(db) or {"status": "no_test_set"}


@app.get("/", include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))
