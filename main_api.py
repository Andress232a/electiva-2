"""
API REST de scouting deportivo (FastAPI + Pydantic).

Carga el paquete models/modelo_scouting_rf.joblib y expone:
    POST /predict   -> predicción (Élite / Estándar) y probabilidad en %
    GET  /salud     -> estado del servicio e información del modelo
    GET  /          -> aplicativo web (index.html)
    GET  /docs      -> documentación interactiva (Swagger UI)

Ejecución (desde la carpeta del proyecto):
    uvicorn main_api:app --reload
"""
import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "models", "modelo_scouting_rf.joblib")
RUTA_INDEX = os.path.join(BASE_DIR, "index.html")

if not os.path.exists(RUTA_MODELO):
    raise RuntimeError(
        "No se encontró models/modelo_scouting_rf.joblib. "
        "Ejecuta primero: python exportar_modelo.py"
    )

# Paquete serializado: modelo + columnas de entrenamiento + metadatos
paquete = joblib.load(RUTA_MODELO)
modelo = paquete["modelo"]
COLUMNAS = paquete["columnas"]
UMBRAL_ELITE_EUR = paquete["umbral_elite_eur"]

app = FastAPI(
    title="API de Scouting Deportivo",
    description=(
        "Predice si un jugador pertenece al 25 % de mayor valor de mercado (Élite) "
        "a partir de su edad, estatura y posición, usando un Random Forest."
    ),
    version="1.0.0",
)

# Permite que index.html (abierto como archivo o desde otro puerto) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Jugador(BaseModel):
    edad: int = Field(..., ge=15, le=45, description="Edad en años (15 a 45).")
    estatura_cm: float = Field(..., ge=150, le=210, description="Estatura en centímetros (150 a 210).")
    posicion: Literal["Attack", "Defender", "Goalkeeper", "Midfield"] = Field(
        ..., description="Posición: Attack, Defender, Goalkeeper o Midfield."
    )

    model_config = {
        "json_schema_extra": {
            "example": {"edad": 21, "estatura_cm": 178, "posicion": "Attack"}
        }
    }


class Prediccion(BaseModel):
    prediccion: Literal["Élite", "Estándar"]
    es_elite: int = Field(..., description="1 = Élite, 0 = Estándar.")
    probabilidad_elite: float = Field(..., description="Probabilidad de ser Élite, en %.")
    probabilidad_estandar: float = Field(..., description="Probabilidad de ser Estándar, en %.")
    umbral_elite_eur: float = Field(..., description="Valor de mercado mínimo para ser Élite (percentil 75).")


@app.post("/predict", response_model=Prediccion, summary="Predecir si un jugador es Élite")
def predict(jugador: Jugador):
    # Misma codificación One-Hot del entrenamiento (la posición base, Attack, queda en ceros)
    fila = {col: 0 for col in COLUMNAS}
    fila["edad"] = jugador.edad
    fila["estatura_cm"] = jugador.estatura_cm
    columna_posicion = f"posicion_{jugador.posicion}"
    if columna_posicion in fila:
        fila[columna_posicion] = 1

    X = pd.DataFrame([fila], columns=COLUMNAS)
    clase = int(modelo.predict(X)[0])
    prob_elite = float(modelo.predict_proba(X)[0][1]) * 100

    return Prediccion(
        prediccion="Élite" if clase == 1 else "Estándar",
        es_elite=clase,
        probabilidad_elite=round(prob_elite, 2),
        probabilidad_estandar=round(100 - prob_elite, 2),
        umbral_elite_eur=UMBRAL_ELITE_EUR,
    )


@app.get("/salud", summary="Estado del servicio")
def salud():
    return {
        "estado": "ok",
        "modelo": type(modelo).__name__,
        "variables": COLUMNAS,
        "hiperparametros": paquete.get("hiperparametros"),
        "metricas_test": paquete.get("metricas_test"),
    }


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(RUTA_INDEX)
