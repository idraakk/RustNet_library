# RustNet - A Tiny, Fast Neural Network in Rust with First‑Class Python Bindings

RustNet is a **from‑scratch, fully‑connected neural network** written in **Rust** and exposed to **Python** via [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs/). It’s intentionally small, readable, and hackable—perfect for learning how modern ML internals work while still getting **native‑speed** inference/training from Rust.

> **Performance report (Colab):**  
> https://colab.research.google.com/drive/1AytQ5TfEe8rPU_ziM7tTPWeXMWD4fPX0?usp=sharing

---

## ✨ Highlights

- **Pure Rust core**: deterministic, fast, memory‑safe implementation.
- **Python‑first UX**: `import rust_net` and go—train from pandas/NumPy in a few lines.
- **Dense MLP**: ReLU hidden layers, **Sigmoid** output for binary classification.
- **Stable training**: BCE‑style gradient at the output (`dZ = A − Y`) + **gradient clipping**.
- **Model persistence**: Save/Load weights & biases as JSON (portable & diff‑friendly).
- **Batteries included**: Example scripts for real data (`run.py`) and **synthetic RF data** (`rf_data_gen.py` + `run2.py`).

---

## 🧠 What’s Inside (Architecture)

- **Layers**: One or more `DenseLayer` blocks with weights `W ∈ R^{in×out}` and bias `b ∈ R^{1×out}`.
- **Activations**:
  - Hidden layers: **ReLU** (`max(0, x)`)
  - Output layer: **Sigmoid** for probabilities in `(0, 1)`
- **Loss/Gradients**:
  - Output layer uses the **cross‑entropy with sigmoid** simplification: `dZ = A − Y` (numerically stable)
  - Hidden layers backprop with `dZ = dA * ReLU'(Z)`
  - **Gradient clipping** on `dW` and `db` to fight NaNs/exploding grads
- **Logging**: We print **MSE** each epoch as a convenient (if imperfect) scalar to track training.

> Aim: Keep the math explicit and the code legible so it’s a great learning/lab tool, not a black box.

---

## 📁 Project Layout

```
rust_net/
├─ Cargo.toml                # Rust crate manifest (builds a Python extension: rust_net)
├─ rf_data_gen.py            # Synthetic RF dataset generator (Frequency, Amplitude, Label)
├─ run.py                    # Example: train & visualize on a CSV (e.g., diabetes.csv)
├─ run2.py                   # Example: train/test on the synthetic RF dataset
└─ src/
   ├─ activations.rs         # ReLU + Sigmoid (and ReLU derivative)
   ├─ data.rs                # CSV → ndarray helpers (optional utility)
   ├─ layers.rs              # Dense layer (W, b) + forward pass
   ├─ lib.rs                 # PyO3 bindings exported as module `rust_net`
   ├─ loss.rs                # MSE (for logging)
   ├─ serde_arrays.rs        # Serialize/deserialize ndarray as JSON
   └─ train.rs               # Network, forward/backward, training loop, save/load
```

---

## 🚀 Quick Start

### 1) Prerequisites

- **Python** 3.8–3.12 (64‑bit recommended)
- **Rust** (via `rustup`) → https://rustup.rs
- **Build tools** (Windows): **Visual Studio Build Tools** with **C++** workload
- **maturin** to build Python extensions

### 2) Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> **PowerShell gotcha:** If activation is blocked, run once per shell:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### 3) Install build/runtime deps

```bash
python -m pip install --upgrade pip
python -m pip install maturin numpy pandas matplotlib
```

### 4) Build & install the Python extension

From the project root (where `Cargo.toml` lives):

```bash
maturin develop
```

This compiles Rust → a Python extension and installs it into the active venv as module **`rust_net`**.

---

## 🧪 Example 1: Train on your CSV (`run.py`)

**Expectations:** your CSV puts **features in every column except the last**, with the **last column as the binary label (0/1)**.

```bash
# Place diabetes.csv next to run.py (or change the filename inside run.py)
python run.py
```

This will:
- Normalize features column‑wise (mean/std)
- Train a tiny net `[n_features, 10, 1]`
- Save `predictions_exported.csv`
- Show a scatter plot of predicted probability vs actual label

---

## 📡 Example 2: Synthetic RF Dataset (`rf_data_gen.py` + `run2.py`)

1) Generate the dataset (1000 points of noisy sine wave):
```bash
python rf_data_gen.py
```
This creates **`rf_data.csv`** with columns: `Frequency`, `Amplitude`, `Label`.

2) Train/test on it:
```bash
python run2.py
```
You’ll get:
- Train/test split (80/20, deterministic)
- Accuracy + confusion matrix on the test set
- `predictions_exported_rf.csv`
- A plot of **predicted probabilities vs actual labels**

---

## 🐍 Python API (Mini‑docs)

```python
import rust_net

# Create a network for binary classification
# Example: 8 input features → 10 hidden units → 1 output prob
nn = rust_net.PyNeuralNetwork([8, 10, 1])

# Train
# inputs: List[List[float]] shape (n, m)
# targets: List[List[float]] shape (n, 1) with 0/1 values
nn.train(inputs, targets, learning_rate=0.001, epochs=2000)

# Predict probabilities (List[List[float]]; each row has length 1)
probs = nn.predict(inputs)

# Save / Load
nn.save("model.json")
nn2 = rust_net.PyNeuralNetwork.load("model.json")
```

**Data shape reminders**
- `inputs` must be **2‑D** (`n_samples × n_features`)
- `targets` must be **2‑D** (`n_samples × 1`) with 0/1 values

**Normalization**  
Always normalize features for stability and faster convergence:
```python
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0
X = (X - mean) / std
```

---

## ⚙️ Design Choices & Rationale

- **Sigmoid + CE gradient** for the final layer (`dZ = A − Y`) is the standard, numerically stable way to train binary classifiers.
- **Gradient clipping** (`[-1, 1]` by default) keeps updates sane on tricky data and guards against NaNs.
- **MSE logging**: We print MSE per epoch for a single scalar view of progress (even though CE is used in backprop). Easy to swap out if you want CE logging instead.
- **JSON checkpoints** keep versioning easy and diffs readable.

Want to go further? Try:
- Mini‑batches & Adam/RMSProp
- Softmax for multi‑class
- Custom activations per layer

---

## 💾 Reproducibility Tips

- For synthetic data: use a fixed RNG seed (see `rf_data_gen.py`).
- Fix train/test split with a seeded RNG (as in `run2.py`).
- Log your hyperparameters (LR, epochs, architecture) alongside results.

---

## 🧯 Troubleshooting

**PowerShell won’t activate my venv**  
Use (per shell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**maturin build errors on Windows**  
Install **Visual Studio Build Tools** with the **C++** workload. Re‑open your shell afterward.

**Module name confusion**  
The Python module is **`rust_net`** (matches `#[pymodule] fn rust_net(...)` in `src/lib.rs`).

**Fresh build**  
If you need to nuke build artifacts:
```bash
cargo clean     # or delete the `target/` folder
maturin develop
```

---

## 🛣️ Roadmap (Ideas)

- Mini‑batch training + optimizers (Adam/RMSProp)
- Multi‑class (softmax + cross‑entropy)
- Pluggable activations per layer
- Optional CE logging & accuracy hooks
- Switchable gradient‑clipping strategies

---

## 🤝 Contributing

PRs and issues are welcome. Keep code readable and well‑commented—this project is both a tool **and** a learning resource. If you add features, include a small example (or test) showing how to use them.

---

## 🪪 License

This project is licensed under the **MIT License**. See the license file for details.
