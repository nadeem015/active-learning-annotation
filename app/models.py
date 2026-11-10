from sqlalchemy import Column,Integer,String,Float,Boolean,DateTime
from datetime import datetime
from .db import Base
class Document(Base):
    __tablename__="documents"
    id=Column(Integer,primary_key=True)
    text=Column(String(5000),nullable=False)
    label=Column(String(100),nullable=True)
    split=Column(String(30),default="pool")
    labeled=Column(Boolean,default=False)
    cycle=Column(Integer,default=0)
class Prediction(Base):
    __tablename__="predictions"
    id=Column(Integer,primary_key=True)
    document_id=Column(Integer)
    cycle=Column(Integer)
    strategy=Column(String(50))
    predicted_label=Column(String(100))
    confidence=Column(Float)
    uncertainty=Column(Float)
class Experiment(Base):
    __tablename__="experiments"
    id=Column(Integer,primary_key=True)
    current_cycle=Column(Integer,default=0)
    strategy=Column(String(50),default="entropy")
    batch_size=Column(Integer,default=5)
    model_version=Column(String(100),default="v0")
    created_at=Column(DateTime,default=datetime.utcnow)
class AnnotationEvent(Base):
    __tablename__="annotation_events"
    id=Column(Integer,primary_key=True)
    document_id=Column(Integer)
    cycle=Column(Integer)
    label=Column(String(100))
    created_at=Column(DateTime,default=datetime.utcnow)
