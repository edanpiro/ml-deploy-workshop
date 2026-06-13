"""
tests/test_api.py – Suite de tests para el servicio de inferencia Iris.

Ejecutar con:
    pytest tests/ -v --tb=short
    pytest tests/ -v --cov=app --cov-report=term-missing   # con cobertura
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

# ─── Fixture ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Cliente de test que reutiliza el mismo lifespan para todos los tests."""
    with TestClient(app) as c:
        yield c


# ─── Constantes de test ───────────────────────────────────────────────────────
SETOSA_SAMPLE    = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VERSICOLOR_SAMPLE = {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}
VIRGINICA_SAMPLE  = {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
CLASSES           = {"setosa", "versicolor", "virginica"}


# ─── Tests: health ────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        """GET /health debe retornar HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_structure(self, client):
        """GET /health debe retornar status, uptime_s y version."""
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert isinstance(data["uptime_s"], (int, float))
        assert isinstance(data["version"], str)


# ─── Tests: predict – casos válidos ──────────────────────────────────────────
class TestPredictValid:
    def test_predict_returns_200(self, client):
        """POST /predict con payload válido debe retornar HTTP 200."""
        resp = client.post("/predict", json=SETOSA_SAMPLE)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        """La respuesta debe tener los campos class_id, class_name, confidence, probabilities."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert "class_id" in data
        assert "class_name" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_class_id_valid(self, client):
        """class_id debe ser 0, 1 o 2."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert data["class_id"] in {0, 1, 2}

    def test_predict_class_name_valid(self, client):
        """class_name debe ser una especie conocida."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert data["class_name"] in CLASSES

    def test_predict_confidence_range(self, client):
        """confidence debe estar entre 0.0 y 1.0."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, client):
        """Las probabilidades de todas las clases deben sumar ~1.0."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 1e-6

    def test_predict_probabilities_keys(self, client):
        """probabilities debe tener las tres especies como llaves."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert set(data["probabilities"].keys()) == CLASSES

    def test_predict_setosa(self, client):
        """Muestra típica de setosa debe predecirse como setosa."""
        data = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert data["class_name"] == "setosa", (
            f"Esperado 'setosa', obtenido '{data['class_name']}'"
        )

    def test_predict_versicolor(self, client):
        """Muestra típica de versicolor debe predecirse como versicolor."""
        data = client.post("/predict", json=VERSICOLOR_SAMPLE).json()
        assert data["class_name"] == "versicolor"

    def test_predict_virginica(self, client):
        """Muestra típica de virginica debe predecirse como virginica."""
        data = client.post("/predict", json=VIRGINICA_SAMPLE).json()
        assert data["class_name"] == "virginica"

    def test_predict_consistency(self, client):
        """La misma entrada siempre debe producir el mismo resultado."""
        r1 = client.post("/predict", json=SETOSA_SAMPLE).json()
        r2 = client.post("/predict", json=SETOSA_SAMPLE).json()
        assert r1["class_id"] == r2["class_id"]
        assert r1["confidence"] == r2["confidence"]


# ─── Tests: predict – casos de error ─────────────────────────────────────────
class TestPredictErrors:
    def test_missing_field_returns_422(self, client):
        """Falta de campos requeridos debe retornar HTTP 422."""
        resp = client.post("/predict", json={"sepal_length": 5.1})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        """Body vacío debe retornar HTTP 422."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_negative_value_returns_422(self, client):
        """Valores negativos (gt=0) deben retornar HTTP 422."""
        bad = {**SETOSA_SAMPLE, "sepal_length": -1.0}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_zero_value_returns_422(self, client):
        """Valor cero (gt=0 no permite 0) debe retornar HTTP 422."""
        bad = {**SETOSA_SAMPLE, "petal_width": 0.0}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_string_value_returns_422(self, client):
        """Strings en campos numéricos deben retornar HTTP 422."""
        bad = {**SETOSA_SAMPLE, "sepal_length": "largo"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_extra_fields_ignored(self, client):
        """Campos extras no deben causar error (FastAPI los ignora)."""
        payload = {**SETOSA_SAMPLE, "campo_extra": 999}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ─── Tests: endpoint raíz ────────────────────────────────────────────────────
class TestRoot:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_message(self, client):
        data = client.get("/").json()
        assert "message" in data
