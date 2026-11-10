from pydantic import BaseModel, Field

from .active_learning import STRATEGIES

StrategyField = Field(default="entropy", pattern="^(" + "|".join(STRATEGIES) + ")$")


class InitRequest(BaseModel):
    seed_size: int = Field(default=12, ge=3, le=500)
    strategy: str = StrategyField
    batch_size: int = Field(default=5, ge=1, le=100)


class QueryRequest(BaseModel):
    strategy: str = StrategyField
    batch_size: int = Field(default=5, ge=1, le=100)


class Annotation(BaseModel):
    document_id: int
    label: str = Field(min_length=1, max_length=100)


class AnnotationBatch(BaseModel):
    annotations: list[Annotation] = Field(min_length=1)
