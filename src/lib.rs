pub mod activations;
pub mod data;
pub mod layers;
pub mod loss;
pub mod train;
pub mod serde_arrays;

use pyo3::prelude::*;
use crate::train::NeuralNetwork;
use ndarray::Array2;

#[pyclass]
pub struct PyNeuralNetwork {
    nn: NeuralNetwork,
}

#[pymethods]
impl PyNeuralNetwork {
    /// Creates a new neural network given a list of layer sizes.
    #[new]
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        PyNeuralNetwork { nn: NeuralNetwork::new(layer_sizes) }
    }

    /// Trains the network.
    /// Expects inputs and targets as lists of lists (2D data).
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, learning_rate: f64, epochs: usize) {
        let n_samples = inputs.len();
        let n_features = if n_samples > 0 { inputs[0].len() } else { 0 };
        let target_cols = if !targets.is_empty() { targets[0].len() } else { 0 };

        let x_flat: Vec<f64> = inputs.into_iter().flatten().collect();
        let t_flat: Vec<f64> = targets.into_iter().flatten().collect();

        let inputs_array = Array2::from_shape_vec((n_samples, n_features), x_flat)
            .expect("Invalid shape for inputs");
        let targets_array = Array2::from_shape_vec((n_samples, target_cols), t_flat)
            .expect("Invalid shape for targets");

        self.nn.train(&inputs_array, &targets_array, learning_rate, epochs);
    }

    /// Returns predictions as a list of lists.
    pub fn predict(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n_samples = inputs.len();
        let n_features = if n_samples > 0 { inputs[0].len() } else { 0 };

        let x_flat: Vec<f64> = inputs.into_iter().flatten().collect();
        let inputs_array = Array2::from_shape_vec((n_samples, n_features), x_flat)
            .expect("Invalid shape for inputs");
        let predictions = self.nn.forward(inputs_array);
        let (rows, _cols) = predictions.dim();
        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            result.push(predictions.slice(ndarray::s![i, ..]).to_vec());
        }
        result
    }

    /// Saves the model to a file.
    pub fn save(&self, path: &str) {
        self.nn.save(path);
    }

    /// Loads a model from a file.
    #[staticmethod]
    pub fn load(path: &str) -> Self {
        let nn = NeuralNetwork::load(path);
        PyNeuralNetwork { nn }
    }
}

#[pymodule]
fn rust_net_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeuralNetwork>()?;
    Ok(())
}
