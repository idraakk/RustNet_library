// ---------------------------------- layers.rs --------------------------------------
// This module defines a single fully-connected (dense) layer:
//   y = x · W + b
// It stores:
//   - weights: (in_features, out_features)
//   - biases:  (1, out_features) broadcast along batch dimension
//   - inputs:  last input batch (for backprop)
//   - outputs: last post-activation outputs (for backprop in our design)

use crate::serde_arrays;               // Custom serde helpers to serialize/deserialize Array2<f64>.
use ndarray::{Array, Array2};          // Array constructor and 2D array type.
use rand::Rng;                         // Random number generation for weight init.
use serde::{Deserialize, Serialize};   // Derive Serialize/Deserialize for checkpointing.

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    #[serde(with = "serde_arrays")]
    pub weights: Array2<f64>,          // Shape: (input_size, output_size)

    #[serde(with = "serde_arrays")]
    pub biases: Array2<f64>,           // Shape: (1, output_size)

    #[serde(skip_serializing, skip_deserializing)]
    pub inputs: Option<Array2<f64>>,   // Last batch input (n, in) used during backprop.

    #[serde(skip_serializing, skip_deserializing)]
    pub outputs: Option<Array2<f64>>,  // Last batch post-activation output (n, out).
}

impl DenseLayer {
    /// Create a new dense layer with random weights in [-0.5, 0.5) and zero biases.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng(); // Random number generator.

        // Initialize weights with small uniform random values to break symmetry.
        let weights = Array::from_shape_fn((input_size, output_size), |_| rng.gen_range(-0.5..0.5));

        // Initialize biases to zeros; (1, out) broadcasts across the batch dimension.
        let biases = Array::zeros((1, output_size));

        DenseLayer {
            weights,
            biases,
            inputs: None,
            outputs: None,
        }
    }

    /// Linear forward pass: Z = X · W + b
    /// Does not apply an activation; the network module decides which activation to apply.
    pub fn forward_linear(&mut self, input: &Array2<f64>) -> Array2<f64> {
        // Cache inputs for gradient computation during backprop.
        self.inputs = Some(input.clone());

        // Matrix multiply (n, in) dot (in, out) = (n, out), then add bias row (broadcasted).
        input.dot(&self.weights) + &self.biases
    }

    /// Store the post-activation outputs A (for backprop and chaining).
    pub fn set_outputs(&mut self, out: Array2<f64>) {
        self.outputs = Some(out);
    }
}
// ------------------------------------------------------------------------------------
