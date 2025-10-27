# ---------------------------------- run.py -----------------------------------------
# This script demonstrates end-to-end usage of the Rust-backed neural network:
# - Loads a CSV with features in all columns except the last (last column = target).
# - Normalizes features.
# - Trains a small network.
# - Exports predictions to CSV and plots predicted vs actual.
#
# The Python module name "rust_net" is produced by our PyO3 #[pymodule] in src/lib.rs.

import rust_net                         # Import the compiled Rust extension module as a Python package.
import pandas as pd                     # Pandas for CSV IO and dataframe utilities.
import numpy as np                      # NumPy for numeric operations (normalization, reshaping).
import matplotlib.pyplot as plt         # Matplotlib for visualization.
import csv                              # csv module for writing a predictions CSV file.

def main() -> None:
    """
    Main pipeline:
      1) Load data from 'diabetes.csv' (or swap to your CSV).
      2) Convert to float64 NumPy arrays (for numeric stability and compatibility).
      3) Normalize features (mean/std per column).
      4) Build a network [n_features, 10, 1] with sigmoid output.
      5) Train using gradient descent (BCE gradient in Rust via dZ = A - Y).
      6) Predict, export results, and plot.
    """
    # 1) Load CSV data. Expect the last column to be the binary target (0/1), all others are features.
    df = pd.read_csv("diabetes.csv")    # You can replace with "rf_data.csv" or your own dataset.
    data = df.values                    # Convert DataFrame to a NumPy array for slicing speed.

    # 2) Split into features (X) and target (y). Ensure 2D shapes (n, m) and (n, 1).
    X = data[:, :-1].astype(np.float64)             # All columns except last -> features (float64).
    y = data[:, -1].reshape(-1, 1).astype(np.float64)  # Last column -> target, reshaped to (n, 1).

    # 3) Normalize features column-wise: (x - mean) / std.
    #    std==0 would cause division by zero; guard by replacing 0 with 1.
    mean = np.mean(X, axis=0)                       # Per-feature mean vector with shape (m,).
    std = np.std(X, axis=0)                         # Per-feature std vector with shape (m,).
    std = np.where(std == 0, 1.0, std)              # Avoid division by zero (constant columns).
    X_norm = (X - mean) / std                       # Elementwise normalization -> shape (n, m).

    # 4) Convert arrays to Python lists of lists so PyO3 can convert to Rust Vec<Vec<f64>>.
    X_list = X_norm.tolist()                        # Converts (n, m) float64 to List[List[float]].
    y_list = y.tolist()                             # Converts (n, 1) float64 to List[List[float]].

    # 5) Instantiate the neural network with input size = num features, hidden=10, output=1.
    #    The Rust side uses ReLU in hidden layers and Sigmoid in the output layer.
    nn = rust_net.PyNeuralNetwork([X_norm.shape[1], 10, 1])

    # 6) Train. Learning rate and epochs are hyperparameters:
    #    - Smaller LR helps stability; more epochs allow convergence.
    #    - Our Rust training uses BCE-style gradient at the output: dZ = A - Y.
    nn.train(X_list, y_list, learning_rate=0.001, epochs=2000)

    # 7) Predict on the training set (for demo). Returns List[List[float]] of probabilities.
    predictions = nn.predict(X_list)

    # 8) Export predictions to CSV for analysis or comparison in Excel/Sheets/etc.
    with open("predictions_exported.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["predicted value", "actual value"])   # Header row
        for pred, actual in zip(predictions, y_list):
            writer.writerow([pred[0], actual[0]])              # Unwrap each 1-element row

    # 9) Prepare lists for plotting. Convert nested lists to flat lists of floats.
    pred_values = [pred[0] for pred in predictions]            # List[float] of predicted probabilities.
    actual_values = [actual[0] for actual in y_list]           # List[float] of ground-truth labels.

    # 10) Plot predicted vs. actual across samples (index on x-axis).
    plt.figure(figsize=(10, 6))                                # Reasonable figure size for readability.
    plt.scatter(range(len(pred_values)), pred_values,
                label="Predicted", alpha=0.7)                  # Predicted probs as dots.
    plt.scatter(range(len(actual_values)), actual_values,
                label="Actual", marker="x", alpha=0.7)         # Actual labels as x's.
    plt.title("Predicted vs Actual Values")                    # Title for the plot.
    plt.xlabel("Sample Index")                                 # X-axis label clarifies index domain.
    plt.ylabel("Value / Probability")                          # Y-axis label (0..1 for probabilities).
    plt.legend()                                               # Show legend to distinguish series.
    plt.grid(True)                                             # Add gridlines for easier reading.
    plt.show()                                                 # Render the plot window.

# Standard Python entrypoint guard: ensures main() only runs when script is called directly.
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------------
