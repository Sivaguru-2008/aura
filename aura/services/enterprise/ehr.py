"""Hospital EHR Database Integration Client."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from aura.common.config import DB_PATH
from aura.schemas.contracts import (
    MultimodalContext,
    LabPanel,
    Symptoms,
    ClinicalHistory,
)


class EHRClient:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the patient clinical records table if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patient_clinical_records (
                    patient_id TEXT PRIMARY KEY,
                    bnp REAL,
                    wbc REAL,
                    procalcitonin REAL,
                    crp REAL,
                    spo2 REAL,
                    fever INTEGER,
                    orthopnea INTEGER,
                    pleuritic_chest_pain INTEGER,
                    dyspnea INTEGER,
                    productive_cough INTEGER,
                    hemoptysis INTEGER,
                    acute_onset INTEGER,
                    prior_cancer INTEGER,
                    smoking_pack_years REAL,
                    copd INTEGER,
                    heart_failure INTEGER,
                    immunosuppression INTEGER
                )
            """)
            conn.commit()

    def get_patient_context(self, patient_id: str) -> Optional[MultimodalContext]:
        """Fetch clinical symptoms and lab results for a given patient ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM patient_clinical_records WHERE patient_id = ?",
                (patient_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            # Construct Labs
            labs = LabPanel(
                bnp=row["bnp"],
                wbc=row["wbc"],
                procalcitonin=row["procalcitonin"],
                crp=row["crp"],
                spo2=row["spo2"]
            )
            
            # Construct Symptoms
            symptoms = Symptoms(
                fever=bool(row["fever"]),
                orthopnea=bool(row["orthopnea"]),
                pleuritic_chest_pain=bool(row["pleuritic_chest_pain"]),
                dyspnea=bool(row["dyspnea"]),
                productive_cough=bool(row["productive_cough"]),
                hemoptysis=bool(row["hemoptysis"]),
                acute_onset=bool(row["acute_onset"])
            )
            
            # Construct History. smoking_pack_years is a non-optional float on the
            # contract, so a NULL column must fall back to the schema default —
            # passing None straight through raises ValidationError, which the
            # pipeline would swallow and silently serve the case with no context.
            history = ClinicalHistory(
                prior_cancer=bool(row["prior_cancer"]),
                smoking_pack_years=float(row["smoking_pack_years"] or 0.0),
                copd=bool(row["copd"]),
                heart_failure=bool(row["heart_failure"]),
                immunosuppression=bool(row["immunosuppression"])
            )
            
            return MultimodalContext(labs=labs, symptoms=symptoms, history=history)

    def save_patient_context(self, patient_id: str, context: MultimodalContext) -> None:
        """Insert or replace a patient clinical record in the EHR database."""
        labs = context.labs
        sym = context.symptoms
        hist = context.history
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO patient_clinical_records (
                    patient_id, bnp, wbc, procalcitonin, crp, spo2,
                    fever, orthopnea, pleuritic_chest_pain, dyspnea,
                    productive_cough, hemoptysis, acute_onset,
                    prior_cancer, smoking_pack_years, copd,
                    heart_failure, immunosuppression
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id, labs.bnp, labs.wbc, labs.procalcitonin, labs.crp, labs.spo2,
                int(sym.fever) if sym.fever is not None else 0,
                int(sym.orthopnea) if sym.orthopnea is not None else 0,
                int(sym.pleuritic_chest_pain) if sym.pleuritic_chest_pain is not None else 0,
                int(sym.dyspnea) if sym.dyspnea is not None else 0,
                int(sym.productive_cough) if sym.productive_cough is not None else 0,
                int(sym.hemoptysis) if sym.hemoptysis is not None else 0,
                int(sym.acute_onset) if sym.acute_onset is not None else 0,
                int(hist.prior_cancer) if hist.prior_cancer is not None else 0,
                hist.smoking_pack_years,
                int(hist.copd) if hist.copd is not None else 0,
                int(hist.heart_failure) if hist.heart_failure is not None else 0,
                int(hist.immunosuppression) if hist.immunosuppression is not None else 0
            ))
            conn.commit()
