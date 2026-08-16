from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from model_engine import RidgeSimulatorEngine
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = RidgeSimulatorEngine()

class InitPayload(BaseModel):
    base_data: List[Dict[str, Any]]
    matriz_data: List[Dict[str, Any]]

class PredictPayload(BaseModel):
    ramo: str
    macro_variables: Dict[str, float]

@app.post("/init_models")
def init_models(payload: InitPayload):
    engine.load_from_json(payload.base_data, payload.matriz_data)
    metrics = engine.train_all_models()
    return {"status": "ok", "entrenados": len(metrics), "metrics": metrics}

@app.post("/predict")
def predict(payload: PredictPayload):
    pred = engine.predict_ramo(payload.ramo, payload.macro_variables)
    return {"prediccion": pred}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)