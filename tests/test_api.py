"""
Unit tests for the FastAPI application.

Tests cover:
- GET method on root endpoint (status code and contents)
- POST method for <=50K prediction
- POST method for >50K prediction
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


class TestGetRoot:
    """Tests for GET / endpoint."""

    def test_get_root_status_code(self, client):
        """Test that GET / returns status code 200."""
        response = client.get("/")
        assert response.status_code == 200, \
            f"Expected status code 200, got {response.status_code}"

    def test_get_root_response_content(self, client):
        """Test that GET / returns expected content."""
        response = client.get("/")
        data = response.json()

        assert "message" in data, "Response should contain 'message' key"
        assert "Welcome" in data["message"], \
            "Message should contain 'Welcome'"


class TestPostPredict:
    """Tests for POST /predict endpoint."""

    def test_predict_income_low(self, client):
        """
        Test prediction for a person likely earning <=50K.

        This tests a sample with characteristics typically associated
        with lower income (young, lower education, etc.)
        """
        # Sample data likely to predict <=50K
        sample_data = {
            "age": 25,
            "workclass": "Private",
            "fnlgt": 226802,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Handlers-cleaners",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 20,
            "native-country": "United-States"
        }

        response = client.post("/predict", json=sample_data)

        assert response.status_code == 200, \
            f"Expected status code 200, got {response.status_code}"

        data = response.json()
        assert "prediction" in data, "Response should contain 'prediction' key"
        assert data["prediction"] == "<=50K", \
            f"Expected prediction '<=50K', got '{data['prediction']}'"

    def test_predict_income_high(self, client):
        """
        Test prediction for a person likely earning >50K.

        This tests a sample with characteristics typically associated
        with higher income (older, higher education, professional job, etc.)
        """
        # Sample data likely to predict >50K
        sample_data = {
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "Doctorate",
            "education-num": 16,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States"
        }

        response = client.post("/predict", json=sample_data)

        assert response.status_code == 200, \
            f"Expected status code 200, got {response.status_code}"

        data = response.json()
        assert "prediction" in data, "Response should contain 'prediction' key"
        assert data["prediction"] == ">50K", \
            f"Expected prediction '>50K', got '{data['prediction']}'"

    def test_predict_status_code(self, client):
        """Test that POST /predict returns status code 200 for valid input."""
        sample_data = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }

        response = client.post("/predict", json=sample_data)

        assert response.status_code == 200, \
            f"Expected status code 200, got {response.status_code}"

    def test_predict_response_contains_prediction(self, client):
        """Test that POST /predict response contains prediction key."""
        sample_data = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }

        response = client.post("/predict", json=sample_data)
        data = response.json()

        assert "prediction" in data, "Response should contain 'prediction' key"
        assert data["prediction"] in ["<=50K", ">50K"], \
            f"Prediction should be '<=50K' or '>50K', got '{data['prediction']}'"

    def test_predict_invalid_input(self, client):
        """Test that POST /predict returns error for invalid input."""
        # Missing required fields
        invalid_data = {
            "age": 39,
            "workclass": "State-gov"
            # Missing other required fields
        }

        response = client.post("/predict", json=invalid_data)

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422, \
            f"Expected status code 422 for invalid input, got {response.status_code}"


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_status_code(self, client):
        """Test that GET /health returns status code 200."""
        response = client.get("/health")
        assert response.status_code == 200, \
            f"Expected status code 200, got {response.status_code}"

    def test_health_response_content(self, client):
        """Test that GET /health returns healthy status."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data, "Response should contain 'status' key"
        assert data["status"] == "healthy", \
            f"Expected status 'healthy', got '{data['status']}'"
