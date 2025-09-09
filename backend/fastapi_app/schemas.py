from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ScoreRow(BaseModel):
    attendance_rate: float = Field(ge=0, le=1)
    avg_score: float
    distance_km: float
    transfers: int
    scholarship_flag: int
    # optional identifiers for logs
    student_id: Optional[str] = None
    school_id: Optional[str] = None

class ScoreRequest(BaseModel):
    rows: List[ScoreRow]

class ScoreResponse(BaseModel):
    proba: List[float]
    band: List[str]

class ExplainRequest(BaseModel):
    rows: List[ScoreRow]
    top_k: int = 5

class ExplainRow(BaseModel):
    student_id: Optional[str]
    contributions: Dict[str, float]  # positive raises risk, negative lowers
    risk_proba: float
    risk_band: str

class ExplainResponse(BaseModel):
    items: List[ExplainRow]

class TrainRequest(BaseModel):
    csv_path: Optional[str] = None  # if None, uses config.DATA_CSV

class MetricsResponse(BaseModel):
    n_samples: int
    class_balance: Dict[str, int]    # {"dropout_0": n0, "dropout_1": n1}
    features: List[str]

class CaseCreate(BaseModel):
    student_id: str
    reason: str
    top_factors: Dict[str, float]

class CaseOut(BaseModel):
    id: int
    student_id: str
    reason: str
    top_factors: Dict[str, float]
    status: str
    owner: str
