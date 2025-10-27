# ---------------------------------- run2.py ----------------------------------------
# This script evaluates the Rust neural network on the SYNTHETIC RF dataset produced
# by a separate generator (rf_data.csv). It:
#   - Loads rf_data.csv (columns: Frequency, Amplitude, Label)
#   - Normalizes features
#   - Splits into train/test
#   - Trains a small network
#   - Computes accuracy and confusion matrix on test set
#   - Saves a predictions CSV and plots probabilities vs labels

import rust_net                             # Our Rust-backed neural network module.
import pandas as pd                         # For reading CSV into a DataFrame.
import numpy as np                          # For numeric ops, normalization, metrics.
import csv                                  # For writing predictions CSV.
import matplotlib.pyplot as plt             # For visualization.
from pathlib import Path                    # For clean, cross-platform file path handling.

def main() -> None:
    """
    End-to-end test flow for the RF synthetic dataset:
      1) Load rf_data.csv with two features and one binary label.
      2) Normalize features.
      3) Split indices into train/test (80/20) deterministically.
      4) Train a network [2, 12, 1] on train set.
      5) Predict on test set, compute accuracy and confusion matrix.
      6) Save predictions and visualize.
    """
    # 1) Verify we have the generated dataset in the current directory.
    data_path = Path("rf_data.csv")
    if not data_path.exists():
        # Give a friendly error if the file is missing, guiding the user to generate it first.
        raise FileNotFoundError(
            "rf_data.csv not found. Generate it with your RF data generator (rf_data_gen.py)."
        )

    # 2) Load the CSV file into memory.
    df = pd.read_csv(data_path)                           # Expect columns: Frequency, Amplitude, Label
    X = df[["Frequency", "Amplitude"]].values.astype(np.float64)  # Shape: (n, 2), features as float64
    y = df["Label"].values.reshape(-1, 1).astype(np.float64)      # Shape: (n, 1), labels as float64 (0/1)

    # 3) Normalize features to have zero mean and unit variance per column.
    mean = np.mean(X, axis=0)                             # Shape: (2,)
    std = np.std(X, axis=0)                               # Shape: (2,)
    std = np.where(std == 0, 1.0, std)                    # Avoid division by zero if a feature is constant.
    X_norm = (X - mean) / std                             # Shape: (n, 2)

    # 4) Create a deterministic train/test split using a fixed RNG seed.
    rng = np.random.default_rng(7)                        # Seed ensures reproducibility across runs.
    n = X_norm.shape[0]                                   # Number of samples
    idx = np.arange(n)                                    # Index array [0, 1, 2, ..., n-1]
    rng.shuffle(idx)                                      # Shuffle indices in-place.
    split = int(0.8 * n)                                  # 80% train, 20% test
    train_idx, test_idx = idx[:split], idx[split:]        # Partition indices.

    X_train, y_train = X_norm[train_idx], y[train_idx]    # Training subset
    X_test,  y_test  = X_norm[test_idx],  y[test_idx]     # Test subset

    # 5) Convert to lists for passing to Rust/PyO3 (Vec<Vec<f64>> interface).
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_test_list  = X_test.tolist()

    # 6) Define a small network. Input size=2 (Frequency, Amplitude), hidden=12, output=1.
    nn = rust_net.PyNeuralNetwork([2, 12, 1])

    # 7) Train using a moderately small learning rate and sufficient epochs for convergence.
    #    Our Rust library applies sigmoid in the output and uses BCE-style gradient (A - Y).
    nn.train(X_train_list, y_train_list, learning_rate=0.003, epochs=3000)

    # 8) Inference on the test set: returns probabilities in range (0, 1).
    test_preds = nn.predict(X_test_list)                      # List[List[float]]
    probs = np.array(test_preds, dtype=np.float64).reshape(-1)  # Shape: (n_test,), flatten nested lists

    # 9) Convert probabilities to class labels with a 0.5 threshold.
    preds = (probs >= 0.5).astype(int).reshape(-1, 1)         # Shape: (n_test, 1)

    # 10) Compute accuracy and confusion matrix elements.
    acc = (preds == y_test).mean()                             # Fraction of correct predictions
    tp = int(((preds == 1) & (y_test == 1)).sum())             # True Positives
    tn = int(((preds == 0) & (y_test == 0)).sum())             # True Negatives
    fp = int(((preds == 1) & (y_test == 0)).sum())             # False Positives
    fn = int(((preds == 0) & (y_test == 1)).sum())             # False Negatives

    # 11) Print metrics in a human-readable format.
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix (test):")
    print(f"  TP: {tp} | FP: {fp}")
    print(f"  FN: {fn} | TN: {tn}")

    # 12) Save predictions to CSV for external analysis and reproducibility.
    with open("predictions_exported_rf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probability", "predicted_label", "actual_label"])
        for p, ytrue in zip(probs, y_test.reshape(-1)):
            w.writerow([float(p), int(p >= 0.5), int(ytrue)])

    # 13) Visualization: compare predicted probabilities vs true labels across the test samples.
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(probs)), probs, label="Predicted Prob(Label=1)", alpha=0.8)
    plt.scatter(range(len(y_test)), y_test.reshape(-1), label="Actual Label", marker="x", alpha=0.8)
    plt.ylim(-0.1, 1.1)                                       # Clamp y-axis to make probs/labels clear.
    plt.title("RF Synthetic — Predicted Probability vs Actual Label (Test Set)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Probability / Label")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------------
