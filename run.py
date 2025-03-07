import rust_net  # or rust_net_py if you kept that name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def main():
    # Load dataset (CSV)
    df = pd.read_csv("diabetes.csv")
    #df = pd.read_csv("rf_data.csv")
    data = df.values
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].reshape(-1, 1).astype(np.float64)

    # Normalize the input features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std

    # Optionally, if your target values are large, consider normalizing them too.
    # For a binary classification problem, targets might be 0 and 1, so they’re fine.

    # Convert arrays to lists for passing to the Rust module
    X_norm_list = X_norm.tolist()
    y_list = y.tolist()

    # Initialize neural network with the number of features, one hidden layer with 10 neurons, and 1 output.
    nn = rust_net.PyNeuralNetwork([X_norm.shape[1], 10, 1])
    
    # Train the network. You might also try a smaller learning rate (e.g., 0.0001) if instability persists.
    nn.train(X_norm_list, y_list, learning_rate=0.01, epochs=5000)
    
    # Predict using the trained model
    predictions = nn.predict(X_norm_list)
    
    # Export predictions to a CSV file with header
    with open("predictions_exported.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["predicted value", "actual value"])
        for pred, actual in zip(predictions, y_list):
            writer.writerow([pred[0], actual[0]])
    
    # Visualization: scatter plot of predicted vs. actual values
    pred_values = [pred[0] for pred in predictions]
    actual_values = [actual[0] for actual in y_list]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(pred_values)), pred_values, color='blue', label='Predicted')
    plt.scatter(range(len(actual_values)), actual_values, color='red', label='Actual', marker='x')
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
