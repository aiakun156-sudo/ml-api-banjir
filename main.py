from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Flood Prediction ML API",
    description="API untuk prediksi banjir menggunakan model Random Forest",
    version="1.0.0"
)

# =========================
# Load ML Model
# =========================
try:
    model = joblib.load("./ml/model_banjir.pkl")
    logger.info(f"✅ Model loaded: {type(model).__name__}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# =========================
# Pydantic Models
# =========================
class SensorData(BaseModel):
    # Terima data dari database yang pakai huruf kecil
    h_kanan: Optional[float] = None
    h_kiri: Optional[float] = None
    q_kanan: Optional[float] = None
    q_kiri: Optional[float] = None
    # Juga support format uppercase untuk backward compatibility
    H_kanan: Optional[float] = None
    H_kiri: Optional[float] = None
    Q_kanan: Optional[float] = None
    Q_kiri: Optional[float] = None

class PredictionResponse(BaseModel):
    status: str
    prediction: int
    confidence: float
    timestamp: str
    recommendation: str
    input_features: dict  # Untuk debugging

# =========================
# Root & Health Check
# =========================
@app.get("/")
async def root():
    return {
        "service": "Flood Prediction ML API",
        "model": "Random Forest Classifier",
        "status": "active" if model else "inactive",
        "endpoint": "POST /predict"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

# =========================
# Predict Endpoint
# =========================
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # =========================
        # Convert ke format model (H dan Q besar)
        # =========================
        input_data = {}
        
        # Priority: ambil dari lowercase (h_kanan, h_kiri, dst)
        # Jika tidak ada, ambil dari uppercase (H_kanan, H_kiri, dst)
        
        h_kanan = data.h_kanan if data.h_kanan is not None else (data.H_kanan or 0.0)
        h_kiri = data.h_kiri if data.h_kiri is not None else (data.H_kiri or 0.0)
        q_kanan = data.q_kanan if data.q_kanan is not None else (data.Q_kanan or 0.0)
        q_kiri = data.q_kiri if data.q_kiri is not None else (data.Q_kiri or 0.0)
        
        # Mapping ke format model (uppercase)
        input_data["H_kanan"] = float(h_kanan)
        input_data["H_kiri"] = float(h_kiri)
        input_data["Q_kanan"] = float(q_kanan)
        input_data["Q_kiri"] = float(q_kiri)
        
        logger.info(f"📥 Input data (normalized): {input_data}")
        
        # Buat DataFrame dengan urutan kolom yang sesuai model
        features = pd.DataFrame([input_data])
        
        logger.info(f"📊 DataFrame columns: {features.columns.tolist()}")
        logger.info(f"📊 DataFrame values: {features.values}")
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Predict confidence
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(proba[int(prediction)]) * 100
        
        logger.info(f"🎯 Prediction: {prediction}, Confidence: {confidence}%")

        # Tentukan status
        status = "BAHAYA" if int(prediction) == 1 else "AMAN"

        # Rekomendasi berdasarkan nilai sensor
        h_kanan_val = float(h_kanan)
        h_kiri_val = float(h_kiri)
        
        if status == "BAHAYA":
            recommendation = "🚨 WASPADA BANJIR! Segera evakuasi dan periksa pintu air."
        elif h_kanan_val > 80 or h_kiri_val > 80:
            recommendation = "⚠️ Ketinggian air meningkat, waspadai potensi banjir."
        else:
            recommendation = "✅ Kondisi normal, tetap pantau ketinggian air."

        return PredictionResponse(
            status=status,
            prediction=int(prediction),
            confidence=round(confidence, 2),
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation,
            input_features=input_data
        )

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"📥 Input data: h_kanan={data.h_kanan}, h_kiri={data.h_kiri}, q_kanan={data.q_kanan}, q_kiri={data.q_kiri}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Debug Endpoint - untuk test format data
# =========================
@app.post("/debug-predict")
async def debug_predict(data: SensorData):
    """
    Endpoint untuk debugging - lihat data yang diterima
    """
    return {
        "received_data": {
            "h_kanan": data.h_kanan,
            "h_kiri": data.h_kiri,
            "q_kanan": data.q_kanan,
            "q_kiri": data.q_kiri,
            "H_kanan": data.H_kanan,
            "H_kiri": data.H_kiri,
            "Q_kanan": data.Q_kanan,
            "Q_kiri": data.Q_kiri,
        },
        "message": "Check yang mana yang ada value-nya"
    }

# Uvicorn akan menjalankan 'app' langsung dari command line
# Jalankan dengan: uvicorn main:app --reload --host 0.0.0.0 --port 8000