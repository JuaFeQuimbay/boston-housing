# app/schemas.py

from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    CRIM: float = Field(..., description="Tasa de criminalidad per cápita")
    ZN: float = Field(..., description="Proporción de suelo residencial")
    INDUS: float = Field(..., description="Proporción de acres comerciales no minoristas")
    CHAS: int = Field(..., description="Cercanía al río Charles: 1 sí, 0 no")
    NOX: float = Field(..., description="Concentración de óxidos de nitrógeno")
    RM: float = Field(..., description="Número promedio de habitaciones")
    AGE: float = Field(..., description="Proporción de viviendas antiguas")
    DIS: float = Field(..., description="Distancia ponderada a centros de empleo")
    RAD: int = Field(..., description="Índice de accesibilidad a autopistas")
    TAX: float = Field(..., description="Tasa de impuestos")
    PTRATIO: float = Field(..., description="Ratio alumno/profesor")
    B: float = Field(..., description="Variable socioeconómica histórica")
    LSTAT: float = Field(..., description="% población de bajo estatus socioeconómico")


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
