"""
Unit tests for the ML model functions.

Tests cover:
- Data processing
- Model training
- Model inference
- Metric computation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data, get_categorical_features
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    compute_metrics_on_slices
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        "age": [39, 50, 38, 53, 28],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private",
                      "Private", "Private"],
        "fnlgt": [77516, 83311, 215646, 234721, 338409],
        "education": ["Bachelors", "Bachelors", "HS-grad",
                      "11th", "Bachelors"],
        "education-num": [13, 13, 9, 7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse",
                           "Divorced", "Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners",
                       "Handlers-cleaners", "Prof-specialty"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family",
                         "Husband", "Wife"],
        "race": ["White", "White", "White", "Black", "Black"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "capital-gain": [2174, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40],
        "native-country": ["United-States", "United-States", "United-States",
                           "United-States", "Cuba"],
        "salary": ["<=50K", "<=50K", "<=50K", "<=50K", "<=50K"]
    })
    return data


@pytest.fixture
def cat_features():
    """Get categorical features."""
    return get_categorical_features()


class TestProcessData:
    """Tests for process_data function."""

    def test_process_data_returns_correct_types(self, sample_data, cat_features):
        """Test that process_data returns correct types."""
        X, y, encoder, lb = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        assert isinstance(X, np.ndarray), "X should be a numpy array"
        assert isinstance(y, np.ndarray), "y should be a numpy array"
        assert encoder is not None, "Encoder should not be None"
        assert lb is not None, "Label Binarizer should not be None"

    def test_process_data_output_shape(self, sample_data, cat_features):
        """Test that process_data returns correct shapes."""
        X, y, encoder, lb = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        assert X.shape[0] == sample_data.shape[0], \
            "Number of samples should match"
        assert y.shape[0] == sample_data.shape[0], \
            "Number of labels should match number of samples"

    def test_process_data_inference_mode(self, sample_data, cat_features):
        """Test process_data in inference mode (training=False)."""
        # First, train
        X_train, y_train, encoder, lb = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        # Then, process in inference mode
        X_test, y_test, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        assert X_train.shape == X_test.shape, \
            "Train and test shapes should match for same data"


class TestTrainModel:
    """Tests for train_model function."""

    def test_train_model_returns_classifier(self, sample_data, cat_features):
        """Test that train_model returns a RandomForestClassifier."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)

        assert isinstance(model, RandomForestClassifier), \
            "Model should be a RandomForestClassifier"

    def test_train_model_is_fitted(self, sample_data, cat_features):
        """Test that the trained model is fitted."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)

        # A fitted model has n_features_in_ attribute
        assert hasattr(model, "n_features_in_"), \
            "Model should be fitted and have n_features_in_ attribute"


class TestInference:
    """Tests for inference function."""

    def test_inference_returns_array(self, sample_data, cat_features):
        """Test that inference returns a numpy array."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)
        preds = inference(model, X)

        assert isinstance(preds, np.ndarray), \
            "Predictions should be a numpy array"

    def test_inference_output_length(self, sample_data, cat_features):
        """Test that inference returns correct number of predictions."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)
        preds = inference(model, X)

        assert len(preds) == len(X), \
            "Number of predictions should match number of inputs"

    def test_inference_binary_output(self, sample_data, cat_features):
        """Test that inference returns binary predictions."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)
        preds = inference(model, X)

        unique_values = np.unique(preds)
        assert all(v in [0, 1] for v in unique_values), \
            "Predictions should be binary (0 or 1)"


class TestComputeModelMetrics:
    """Tests for compute_model_metrics function."""

    def test_compute_metrics_returns_tuple(self):
        """Test that compute_model_metrics returns a tuple of 3 values."""
        y = np.array([0, 1, 1, 0, 1])
        preds = np.array([0, 1, 0, 0, 1])

        result = compute_model_metrics(y, preds)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 3, "Result should have 3 values"

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y = np.array([0, 1, 1, 0, 1])
        preds = np.array([0, 1, 1, 0, 1])

        precision, recall, fbeta = compute_model_metrics(y, preds)

        assert precision == 1.0, "Precision should be 1.0 for perfect predictions"
        assert recall == 1.0, "Recall should be 1.0 for perfect predictions"
        assert fbeta == 1.0, "F1 should be 1.0 for perfect predictions"

    def test_compute_metrics_returns_floats(self):
        """Test that compute_model_metrics returns floats."""
        y = np.array([0, 1, 1, 0, 1])
        preds = np.array([0, 1, 0, 1, 1])

        precision, recall, fbeta = compute_model_metrics(y, preds)

        assert isinstance(precision, float), "Precision should be a float"
        assert isinstance(recall, float), "Recall should be a float"
        assert isinstance(fbeta, float), "F1 should be a float"

    def test_compute_metrics_range(self):
        """Test that metrics are in valid range [0, 1]."""
        y = np.array([0, 1, 1, 0, 1])
        preds = np.array([0, 1, 0, 1, 1])

        precision, recall, fbeta = compute_model_metrics(y, preds)

        assert 0 <= precision <= 1, "Precision should be between 0 and 1"
        assert 0 <= recall <= 1, "Recall should be between 0 and 1"
        assert 0 <= fbeta <= 1, "F1 should be between 0 and 1"


class TestSliceMetrics:
    """Tests for slice metrics computation."""

    def test_slice_metrics_returns_list(self, sample_data, cat_features):
        """Test that compute_metrics_on_slices returns a list."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)
        preds = inference(model, X)

        # Reset index for alignment
        sample_data_reset = sample_data.reset_index(drop=True)

        metrics = compute_metrics_on_slices(
            sample_data_reset, "sex", y, preds, cat_features
        )

        assert isinstance(metrics, list), "Result should be a list"
        assert len(metrics) > 0, "Result should not be empty"

    def test_slice_metrics_contains_required_keys(self, sample_data, cat_features):
        """Test that slice metrics contain all required keys."""
        X, y, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        model = train_model(X, y)
        preds = inference(model, X)

        sample_data_reset = sample_data.reset_index(drop=True)

        metrics = compute_metrics_on_slices(
            sample_data_reset, "sex", y, preds, cat_features
        )

        required_keys = {"feature", "value", "n_samples",
                         "precision", "recall", "fbeta"}

        for metric in metrics:
            assert required_keys.issubset(metric.keys()), \
                f"Metric should contain all required keys: {required_keys}"
