# app/model_loader.py

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


logger = logging.getLogger(__name__)

# Resuelve rutas relativas a la raíz del proyecto, sin depender del cwd
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DEFAULT_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# Orden por defecto si el modelo se entrenó con una versión sin metadata
DEFAULT_FEATURE_ORDER: List[str] = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]


@dataclass
class ModelBundle:
    """Modelo entrenado junto con su metadata de trazabilidad."""

    model: Any
    version: str
    feature_order: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _load_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        logger.warning(
            "model_metadata.json not found at %s — falling back to defaults",
            metadata_path,
        )
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Failed to parse model_metadata.json")
        return {}


def load_model(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> ModelBundle:
    """
    Carga el modelo entrenado y su metadata asociada.

    Si la metadata no existe, devuelve valores por defecto consistentes
    con el dataset Boston Housing. La aplicación sigue funcionando
    pero con menor trazabilidad.
    """

    model_path = model_path or DEFAULT_MODEL_PATH
    metadata_path = metadata_path or DEFAULT_METADATA_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta primero: python -m src.train"
        )

    model = joblib.load(model_path)
    metadata = _load_metadata(metadata_path)

    return ModelBundle(
        model=model,
        version=metadata.get("model_version", "unknown"),
        feature_order=metadata.get("feature_order", DEFAULT_FEATURE_ORDER),
        metadata=metadata,
    )
