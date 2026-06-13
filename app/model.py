"""
app/model.py – Carga el modelo entrenado y expone la lógica de predicción.
"""
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

MODEL_PATH = Path("models/iris_model.pkl")
CLASSES = ["setosa", "versicolor", "virginica"]


class IrisModel:
    """Wrapper sobre el pipeline sklearn para servir predicciones."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}. "
                "Ejecuta primero: python train.py"
            )
        self._pipeline = joblib.load(model_path)

    def predict(self, features: list) -> Dict[str, Any]:
        """
        Recibe una lista de listas con features y retorna dict con:
          - class_id: índice de la clase predicha
          - class_name: nombre de la clase
          - confidence: probabilidad de la clase predicha
          - probabilities: probabilidades de todas las clases
        """
        arr = np.array(features, dtype=float)
        class_ids = self._pipeline.predict(arr)
        probas = self._pipeline.predict_proba(arr)

        # Tomamos el primer ejemplo (batch de 1)
        class_id = int(class_ids[0])
        confidence = float(probas[0][class_id])

        return {
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "confidence": confidence,
            "probabilities": {
                cls: float(probas[0][i])
                for i, cls in enumerate(CLASSES)
            },
        }
