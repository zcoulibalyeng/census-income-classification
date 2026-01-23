"""
Model training, inference, and evaluation functions for Census Income Classification.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, filepath):
    """
    Save a trained model to disk.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    filepath : str
        Path where the model should be saved.
    """
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a trained model from disk.

    Inputs
    ------
    filepath : str
        Path to the saved model.

    Returns
    -------
    model : sklearn model
        Loaded machine learning model.
    """
    model = joblib.load(filepath)
    return model


def save_encoder(encoder, filepath):
    """
    Save a trained encoder to disk.

    Inputs
    ------
    encoder : sklearn encoder
        Trained encoder (OneHotEncoder or LabelBinarizer).
    filepath : str
        Path where the encoder should be saved.
    """
    joblib.dump(encoder, filepath)


def load_encoder(filepath):
    """
    Load a trained encoder from disk.

    Inputs
    ------
    filepath : str
        Path to the saved encoder.

    Returns
    -------
    encoder : sklearn encoder
        Loaded encoder.
    """
    encoder = joblib.load(filepath)
    return encoder


def compute_metrics_on_slices(
    data,
    feature,
    y,
    preds,
    categorical_features
):
    """
    Compute performance metrics on slices of the data for a given feature.

    Inputs
    ------
    data : pd.DataFrame
        Original dataframe (before processing).
    feature : str
        Name of the feature to slice on.
    y : np.ndarray
        True labels.
    preds : np.ndarray
        Predicted labels.
    categorical_features : list[str]
        List of categorical feature names.

    Returns
    -------
    slice_metrics : list[dict]
        List of dictionaries containing metrics for each slice.
    """
    slice_metrics = []

    # Get unique values for the feature
    unique_values = data[feature].unique()

    for value in unique_values:
        # Get indices where feature equals value
        mask = data[feature] == value

        if mask.sum() == 0:
            continue

        # Get y and preds for this slice
        y_slice = y[mask]
        preds_slice = preds[mask]

        # Compute metrics
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

        slice_metrics.append({
            "feature": feature,
            "value": value,
            "n_samples": int(mask.sum()),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        })

    return slice_metrics


def compute_metrics_all_slices(data, y, preds, categorical_features):
    """
    Compute performance metrics on slices for all categorical features.

    Inputs
    ------
    data : pd.DataFrame
        Original dataframe (before processing).
    y : np.ndarray
        True labels.
    preds : np.ndarray
        Predicted labels.
    categorical_features : list[str]
        List of categorical feature names.

    Returns
    -------
    all_metrics : list[dict]
        List of dictionaries containing metrics for all slices.
    """
    all_metrics = []

    for feature in categorical_features:
        slice_metrics = compute_metrics_on_slices(
            data, feature, y, preds, categorical_features
        )
        all_metrics.extend(slice_metrics)

    return all_metrics


def write_slice_metrics_to_file(metrics, filepath):
    """
    Write slice metrics to a text file.

    Inputs
    ------
    metrics : list[dict]
        List of dictionaries containing metrics for each slice.
    filepath : str
        Path to the output file.
    """
    with open(filepath, "w") as f:
        f.write("Model Performance on Data Slices\n")
        f.write("=" * 80 + "\n\n")

        current_feature = None
        for metric in metrics:
            if metric["feature"] != current_feature:
                current_feature = metric["feature"]
                f.write(f"\nFeature: {current_feature}\n")
                f.write("-" * 40 + "\n")

            f.write(f"  Value: {metric['value']}\n")
            f.write(f"    Samples: {metric['n_samples']}\n")
            f.write(f"    Precision: {metric['precision']:.4f}\n")
            f.write(f"    Recall: {metric['recall']:.4f}\n")
            f.write(f"    F1 (Fbeta): {metric['fbeta']:.4f}\n")
