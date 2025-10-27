// ----------------------------------- lib.rs ----------------------------------------
// This is the Rust <-> Python bridge via PyO3.
// It exposes a Python class `PyNeuralNetwork` with methods:
//   - __init__(layer_sizes)
//   - train(inputs, targets, learning_rate, epochs)
//   - predict(inputs)
//   - save(path)
//   - load(path) -> PyNeuralNetwork
//
// The module is exported to Python under the name "rust_net" (matching Cargo.toml crate name).

pub mod activations;   // Internal modules used by the network.
pub mod data;
pub mod layers;
pub mod loss;
pub mod train;
pub mod serde_arrays;

use crate::train::NeuralNetwork;   // The core Rust neural network implementation.
use ndarray::Array2;               // Used to construct arrays from Python inputs.
use pyo3::prelude::*;              // PyO3 prelude: macros/traits for Python bindings.

/// Python-visible wrapper around the Rust NeuralNetwork.
#[pyclass]
pub struct PyNeuralNetwork {
    nn: NeuralNetwork,             // Composition: our Rust model.
}

#[pymethods]
impl PyNeuralNetwork {
    /// __init__(self, layer_sizes: List[int])
    /// layer_sizes example: [n_features, 10, 1]
    #[new]
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        PyNeuralNetwork {
            nn: NeuralNetwork::new(layer_sizes),
        }
    }

    /// Train the network on 2D lists.
    /// `inputs`:  shape (n, m) -> List[List[float]]
    /// `targets`: shape (n, k) -> List[List[float]] (k=1 for binary)
    pub fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        learning_rate: f64,
        epochs: usize,
    ) {
        // Derive sizes from Python lists (outer length = n_samples).
        let n_samples = inputs.len();
        let n_features = if n_samples > 0 { inputs[0].len() } else { 0 };
        let target_cols = if !targets.is_empty() { targets[0].len() } else { 0 };

        // Flatten nested lists into a contiguous Vec<f64> (row-major order).
        let x_flat: Vec<f64> = inputs.into_iter().flatten().collect();
        let t_flat: Vec<f64> = targets.into_iter().flatten().collect();

        // Reconstruct 2D arrays that the Rust NN expects.
        let inputs_array =
            Array2::from_shape_vec((n_samples, n_features), x_flat).expect("Invalid inputs shape");
        let targets_array =
            Array2::from_shape_vec((n_samples, target_cols), t_flat).expect("Invalid targets shape");

        // Delegate to Rust training loop.
        self.nn
            .train(&inputs_array, &targets_array, learning_rate, epochs);
    }

    /// Predict outputs for given 2D list inputs. Returns a 2D list (n, k).
    pub fn predict(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n_samples = inputs.len();
        let n_features = if n_samples > 0 { inputs[0].len() } else { 0 };

        let x_flat: Vec<f64> = inputs.into_iter().flatten().collect();
        let inputs_array =
            Array2::from_shape_vec((n_samples, n_features), x_flat).expect("Invalid inputs shape");

        // Run forward pass; returns (n, k) probabilities for binary k=1.
        let predictions = self.nn.forward(inputs_array);

        // Convert ndarray back to Vec<Vec<f64>> for Python.
        let (rows, _cols) = predictions.dim();
        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            result.push(predictions.slice(ndarray::s![i, ..]).to_vec());
        }
        result
    }

    /// Save model weights/biases to a JSON file at `path`.
    pub fn save(&self, path: &str) {
        self.nn.save(path);
    }

    /// Load a model from JSON at `path` and return a new PyNeuralNetwork instance.
    #[staticmethod]
    pub fn load(path: &str) -> Self {
        let nn = NeuralNetwork::load(path);
        PyNeuralNetwork { nn }
    }
}

/// Module initialization function. Exports `PyNeuralNetwork` under the module `rust_net`.
#[pymodule]
fn rust_net(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeuralNetwork>()?;
    Ok(())
}
// ------------------------------------------------------------------------------------
