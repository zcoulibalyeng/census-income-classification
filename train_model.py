"""
Script to train machine learning model on Census Income data.

This script:
1. Loads and processes the census data
2. Trains a RandomForest classifier
3. Evaluates performance on test data
4. Computes and outputs slice metrics
5. Saves the model and encoders
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data, get_categorical_features
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    save_encoder,
    compute_metrics_all_slices,
    write_slice_metrics_to_file
)


def main():
    """Main function to train and evaluate the model."""

    # Define paths
    data_path = "data/census.csv"
    model_dir = "model"
    model_path = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    lb_path = os.path.join(model_dir, "lb.pkl")
    slice_output_path = "slice_output.txt"

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    # Get categorical features
    cat_features = get_categorical_features()
    print(f"Categorical features: {cat_features}")

    # Split data
    print("\nSplitting data into train/test sets...")
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Process training data
    print("\nProcessing training data...")
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Process test data
    print("\nProcessing test data...")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    print("Model training complete.")

    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_preds = inference(model, X_train)
    train_precision, train_recall, train_fbeta = compute_model_metrics(
        y_train, train_preds
    )
    print(f"Train Precision: {train_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f}")
    print(f"Train F1: {train_fbeta:.4f}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_preds = inference(model, X_test)
    test_precision, test_recall, test_fbeta = compute_model_metrics(
        y_test, test_preds
    )
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_fbeta:.4f}")

    # Compute slice metrics on test data
    print("\nComputing metrics on data slices...")
    # Reset test index to align with predictions
    test_reset = test.reset_index(drop=True)
    slice_metrics = compute_metrics_all_slices(
        test_reset, y_test, test_preds, cat_features
    )
    write_slice_metrics_to_file(slice_metrics, slice_output_path)
    print(f"Slice metrics written to: {slice_output_path}")

    # Save model and encoders
    print("\nSaving model and encoders...")
    save_model(model, model_path)
    save_encoder(encoder, encoder_path)
    save_encoder(lb, lb_path)
    print(f"Model saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Label Binarizer saved to: {lb_path}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    # Summary
    print("\n--- Model Performance Summary ---")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1 Score:  {test_fbeta:.4f}")


if __name__ == "__main__":
    main()
