from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from .database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    school_id = Column(String, index=True)
    payload = Column(JSON)
    risk_proba = Column(Float)
    risk_band = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    reason = Column(String)        # short human-readable reason
    top_factors = Column(JSON)     # e.g., {"attendance_rate": 0.32, "avg_score": 0.21}
    status = Column(String, default="OPEN")  # OPEN | IN_PROGRESS | CLOSED
    owner = Column(String, default="system")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
