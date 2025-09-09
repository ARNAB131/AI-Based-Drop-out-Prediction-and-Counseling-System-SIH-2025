import os
import io
import json
import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from .config import get_settings
from .schemas import (
    ScoreRequest, ScoreResponse, ExplainRequest, ExplainResponse, ExplainRow,
    TrainRequest, MetricsResponse, CaseCreate, CaseOut
)
from .utils.model_utils import train_or_load, predict_with_bands, explain_rows, train_from_csv
from .database import SessionLocal, init_db
from .models_db import PredictionLog, Case

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dropout-api")

settings = get_settings()

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --------------- DB Session dependency ---------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------- Bootstrap: DB + Model ---------------
init_db()
model = train_or_load(
    model_path=os.path.join(os.path.dirname(__file__), settings.MODEL_PATH),
    data_csv=os.path.join(os.path.dirname(__file__), settings.DATA_CSV),
    feats=settings.FEATS,
)

@app.get("/health")
def health():
    return {"status": "ok", "version": settings.APP_VERSION}

# --------------- Train ----------------
@app.post("/train")
def train(req: TrainRequest) -> MetricsResponse:
    csv_path = req.csv_path or os.path.join(os.path.dirname(__file__), settings.DATA_CSV)
    model_path = os.path.join(os.path.dirname(__file__), settings.MODEL_PATH)
    new_model, metrics = train_from_csv(model_path, csv_path, settings.FEATS)
    global model
    model = new_model
    return MetricsResponse(
        n_samples=metrics["n_samples"],
        class_balance=metrics["class_balance"],
        features=settings.FEATS
    )

# --------------- Score (JSON) ----------------
@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, db: Session = Depends(get_db)):
    df = pd.DataFrame([r.dict() for r in req.rows])
    missing = [c for c in settings.FEATS if c not in df.columns]
    if missing:
        raise HTTPException(400, detail=f"missing columns: {missing}")
    proba, bands = predict_with_bands(model, df[settings.FEATS])

    # optional logging
    for i, row in df.iterrows():
        try:
            db.add(PredictionLog(
                student_id=row.get("student_id"), school_id=row.get("school_id"),
                payload=row[settings.FEATS].to_dict(), risk_proba=float(proba[i]), risk_band=bands[i]
            ))
        except Exception as e:
            log.warning(f"log insert failed: {e}")
    db.commit()

    return ScoreResponse(proba=[float(p) for p in proba], band=bands)

# --------------- Score (CSV upload) ----------------
@app.post("/score_csv")
async def score_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    missing = [c for c in settings.FEATS if c not in df.columns]
    if missing:
        raise HTTPException(400, detail=f"missing columns: {missing}")
    proba, bands = predict_with_bands(model, df[settings.FEATS])
    df_out = df.copy()
    df_out["risk_proba"] = proba
    df_out["risk_band"] = bands

    # optional logging
    for i, row in df_out.iterrows():
        try:
            db.add(PredictionLog(
                student_id=row.get("student_id"), school_id=row.get("school_id"),
                payload=row[settings.FEATS].to_dict(), risk_proba=float(row["risk_proba"]), risk_band=row["risk_band"]
            ))
        except Exception as e:
            log.warning(f"log insert failed: {e}")
    db.commit()

    return json.loads(df_out.to_json(orient="records"))

# --------------- Explain (SHAP per-row) ----------------
@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    df = pd.DataFrame([r.dict() for r in req.rows])
    missing = [c for c in settings.FEATS if c not in df.columns]
    if missing:
        raise HTTPException(400, detail=f"missing columns: {missing}")
    proba, bands = predict_with_bands(model, df[settings.FEATS])
    contribs = explain_rows(model, df[settings.FEATS], top_k=req.top_k)

    items: List[ExplainRow] = []
    for i, r in enumerate(req.rows):
        items.append(ExplainRow(
            student_id=r.student_id,
            contributions=contribs[i],
            risk_proba=float(proba[i]),
            risk_band=bands[i]
        ))
    return ExplainResponse(items=items)

# --------------- Basic dataset metrics (from training CSV) ---------------
@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    csv_path = os.path.join(os.path.dirname(__file__), settings.DATA_CSV)
    df = pd.read_csv(csv_path)
    y = df["dropout_flag"].astype(int)
    return MetricsResponse(
        n_samples=int(len(df)),
        class_balance={"dropout_0": int((y==0).sum()), "dropout_1": int((y==1).sum())},
        features=settings.FEATS
    )

# --------------- Minimal case endpoints ---------------
@app.post("/cases", response_model=CaseOut)
def create_case(payload: CaseCreate, db: Session = Depends(get_db)):
    c = Case(student_id=payload.student_id, reason=payload.reason, top_factors=payload.top_factors)
    db.add(c); db.commit(); db.refresh(c)
    return CaseOut(id=c.id, student_id=c.student_id, reason=c.reason, top_factors=c.top_factors, status=c.status, owner=c.owner)

@app.get("/cases/{student_id}", response_model=List[CaseOut])
def list_cases(student_id: str, db: Session = Depends(get_db)):
    rows = db.query(Case).filter(Case.student_id==student_id).order_by(Case.id.desc()).all()
    return [CaseOut(id=r.id, student_id=r.student_id, reason=r.reason, top_factors=r.top_factors, status=r.status, owner=r.owner) for r in rows]
