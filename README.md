# RustNet - A Rust-based Neural Network Library with Python Bindings

RustNet is a simple, fully-connected (dense) neural network library written in Rust, with Python bindings via **PyO3**.  
It allows you to define, train, and use a neural network directly in Python while leveraging Rust's performance advantages.

---

## 📌 Features
- **Custom Neural Network Implementation** – Not based on existing ML frameworks.
- **Supports Binary Classification** – Using **sigmoid activation** in the final layer.
- **Gradient Descent with Backpropagation** – Includes **gradient clipping** to prevent numerical instability.
- **Training and Prediction Support** – Load data, normalize it, train, and make predictions.
- **Seamless Python Integration** – Use it just like any other Python package.

---

## 📂 Project Structure
```
rust_net/
├── Cargo.toml               # Rust package metadata and dependencies.
└── src/
    ├── activations.rs       # Activation functions (ReLU, Sigmoid, etc.)
    ├── data.rs              # CSV data loading utilities.
    ├── layers.rs            # Dense layer implementation.
    ├── loss.rs              # Loss function (MSE).
    ├── serde_arrays.rs      # Custom serialization for ndarrays.
    ├── train.rs             # Training (forward/backprop, gradient clipping).
    ├── lib.rs               # Library entry point with PyO3 bindings.
└── run.py                   # Example usage in Python.
```

---

## 🛠 Installation Guide

### 1️⃣ Prerequisites
Before you begin, ensure that you have the following installed:
- **Rust** (via [rustup](https://rustup.rs)):  
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Python (3.7+)**
- **Virtual Environment** (Recommended)
- **Maturin** (For building Rust Python extensions):
  ```bash
  pip install maturin
  ```

### 2️⃣ Create & Activate a Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3️⃣ Build the Library and Install it
Navigate to the project root (where `Cargo.toml` is located) and run:
```bash
maturin develop
```
OR, if you prefer to build a wheel:
```bash
maturin build
pip install target/wheels/rust_net-0.1.0-cp39-cp39-win_amd64.whl
```

### 4️⃣ Verify Installation
```bash
pip show rust_net
```
Or test in Python:
```python
import rust_net_py
print(rust_net_py.__file__)
```

---

## 🚀 Usage Example (Python)
Below is an example script (**run.py**) to train and use the RustNet neural network in Python.

```python
import rust_net_py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def main():
    df = pd.read_csv("data.csv")  # Load dataset.
    data = df.values  
    X = data[:, :-1]  # Features.
    y = data[:, -1].reshape(-1, 1)  # Targets.

    X = X.astype(np.float64)
    y = y.astype(np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std

    X_norm_list = X_norm.tolist()
    y_list = y.tolist()

    nn = rust_net_py.PyNeuralNetwork([len(X_norm[0]), 10, 1])
    nn.train(X_norm_list, y_list, learning_rate=0.0001, epochs=1000)

    predictions = nn.predict(X_norm_list)

    with open("predictions_exported.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["predicted value", "actual value"])
        for pred, actual in zip(predictions, y_list):
            writer.writerow([pred[0], actual[0]])

    pred_values = [pred[0] for pred in predictions]
    actual_values = [actual[0] for actual in y_list]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(pred_values)), pred_values, color='blue', label='Predicted", alpha=0.7)
    plt.scatter(range(len(actual_values)), actual_values, color='red', label='Actual", marker='x', alpha=0.7)
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 📊 Visualizing the Predictions
Once the script runs, a scatter plot will be generated comparing **Predicted Values (Blue)** vs **Actual Values (Red)**.

---

## 🎯 Summary of Steps
1. **Install Dependencies:** Rust, Python, Virtual Environment, Maturin.
2. **Build the Rust Library:** Run `maturin develop`.
3. **Verify Installation:** Test import in Python.
4. **Train & Predict:** Use `run.py` with your dataset.
5. **Export & Visualize Predictions:** A CSV file and plot are generated.

---

## 🏗 Future Improvements
- Multi-class classification support.
- Support for different activation functions in hidden layers.
- Implement additional optimization techniques.

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit an issue or pull request.

---

## 📝 License
This project is licensed under the MIT License.
