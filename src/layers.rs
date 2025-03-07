use ndarray::{Array, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::serde_arrays;

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    #[serde(with = "serde_arrays")]
    pub weights: Array2<f64>,
    #[serde(with = "serde_arrays")]
    pub biases: Array2<f64>,
    #[serde(skip_serializing, skip_deserializing)]
    pub inputs: Option<Array2<f64>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub outputs: Option<Array2<f64>>,
}

impl DenseLayer {
    /// Creates a new dense layer with random weights and zero biases.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn((input_size, output_size), |_| rng.gen_range(-0.5..0.5));
        let biases = ndarray::Array::zeros((1, output_size));
        DenseLayer { weights, biases, inputs: None, outputs: None }
    }

    /// Computes the forward pass for the layer.
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.inputs = Some(input.clone());
        input.dot(&self.weights) + &self.biases
    }
}
