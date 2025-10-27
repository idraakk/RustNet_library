// ----------------------------------- train.rs --------------------------------------
// This module defines the core NeuralNetwork struct and its training loop.
// Design choices:
//  - Hidden layers use ReLU; output layer uses Sigmoid.
//  - Output layer gradient uses BCE simplification: dZ = A - Y (with Sigmoid + CE).
//  - Gradient clipping prevents exploding updates and NaNs in weights/biases.
//  - We log MSE for a rough convergence signal and export predictions at last epoch.

use crate::activations::{relu, relu_derivative, sigmoid}; // Activation functions.
use crate::layers::DenseLayer;                            // Dense layer definition.
use crate::loss::mean_squared_error;                      // For logging purposes.
use ndarray::{Array1, Array2, Axis};                      // Array types and axis utility.
use serde::{Deserialize, Serialize};                      // For saving/loading the network.
use std::fs::File;                                        // File IO for model/predictions.
use std::io::{BufWriter, Write};                          // Buffered writer for CSV export.

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,       // Ordered list of layers: [input->hidden1, hidden1->hidden2, ..., last]
}

/// Clip helper for 2D arrays: limits absolute values to +/- limit in-place.
fn clip2d(arr: &mut Array2<f64>, limit: f64) {
    arr.mapv_inplace(|x| {
        if x > limit {
            limit
        } else if x < -limit {
            -limit
        } else {
            x
        }
    });
}

/// Clip helper for 1D arrays: limits absolute values to +/- limit in-place.
fn clip1d(arr: &mut Array1<f64>, limit: f64) {
    arr.mapv_inplace(|x| {
        if x > limit {
            limit
        } else if x < -limit {
            -limit
        } else {
            x
        }
    });
}

impl NeuralNetwork {
    /// Construct a new MLP from layer sizes, e.g., [n_in, h1, h2, ..., n_out].
    /// Panics if fewer than 2 sizes are provided (must have input and output).
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least [input_size, output_size]"
        );

        // Create dense layers connecting consecutive sizes: (sizes[i] -> sizes[i+1]).
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        NeuralNetwork { layers }
    }

    /// Forward pass through all layers:
    ///   Z = X·W + b
    ///   A = ReLU(Z) for hidden layers; A = Sigmoid(Z) for the final layer.
    /// Returns the final activations (predicted probabilities for binary).
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        let last = self.layers.len().saturating_sub(1); // Index of last layer.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let z = layer.forward_linear(&input);           // Linear transform: X·W + b
            let a = if i == last { sigmoid(&z) } else {     // Apply activation
                relu(&z)
            };
            layer.set_outputs(a.clone());                   // Cache post-activation for backprop.
            input = a;                                      // Feed to next layer.
        }
        input
    }

    /// Backpropagation step:
    ///   - Output layer gradient (sigmoid + CE): dZ = A - Y
    ///   - Hidden layers: dZ = dA * ReLU'(Z). We compute ReLU' from post-activation.
    ///   - Parameter updates: W -= α * dW, b -= α * db
    ///   - Gradient clipping before updates.
    pub fn backward(&mut self, targets: &Array2<f64>, learning_rate: f64) {
        let num_layers = self.layers.len();
        let mut gradient: Option<Array2<f64>> = None; // Gradient wrt layer outputs dA for next iteration.
        let clip_val = 1.0;                            // Threshold for gradient clipping.

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            // Ensure we have cached values from forward pass.
            let outputs = layer
                .outputs
                .as_ref()
                .expect("Missing layer outputs (forward not run)");
            let inputs = layer
                .inputs
                .as_ref()
                .expect("Missing layer inputs (forward not run)");

            // Batch size as f64 for averaging gradients.
            let m = inputs.dim().0 as f64;

            if i == num_layers - 1 {
                // ----------------------- Output layer (Sigmoid + CE) -----------------------
                // For sigmoid output with binary cross-entropy, the derivative simplifies:
                //   dZ = A - Y  (where A is post-sigmoid outputs, Y in {0,1})
                let d_z = outputs - targets;                     // (n, out)
                let mut d_w = inputs.t().dot(&d_z) / m;          // (in, out)
                let mut db = d_z.sum_axis(Axis(0)) / m;          // (out,)

                // Clip gradients to prevent exploding updates / NaNs.
                clip2d(&mut d_w, clip_val);
                clip1d(&mut db, clip_val);

                // Compute gradient wrt previous activation: dA_prev = dZ · W^T (using OLD weights).
                let next_grad = d_z.dot(&layer.weights.t());     // (n, in)

                // Parameter updates: W := W - α * dW; b := b - α * db
                layer.weights -= &(d_w * learning_rate);
                // db is 1D (out,). Convert to (1, out) owned array then scale.
                let db2d = db.view().insert_axis(Axis(0)).to_owned();
                layer.biases -= &(db2d * learning_rate);

                gradient = Some(next_grad);                      // Push gradient to next layer (backwards).
            } else {
                // --------------------------- Hidden layer (ReLU) ---------------------------
                // dZ = dA * ReLU'(Z). We use post-activation to derive ReLU' safely.
                let d_a = gradient.take().expect("Missing gradient");
                let d_z = d_a * relu_derivative(outputs);        // Elementwise product (n, out)

                let mut d_w = inputs.t().dot(&d_z) / m;          // (in, out)
                let mut db = d_z.sum_axis(Axis(0)) / m;          // (out,)

                clip2d(&mut d_w, clip_val);
                clip1d(&mut db, clip_val);

                let next_grad = d_z.dot(&layer.weights.t());     // (n, in)

                layer.weights -= &(d_w * learning_rate);
                let db2d = db.view().insert_axis(Axis(0)).to_owned();
                layer.biases -= &(db2d * learning_rate);

                gradient = Some(next_grad);
            }
        }
    }

    /// Training loop:
    ///  - Runs forward/backward for `epochs`.
    ///  - Logs MSE for a sense of progress (even though CE grad used).
    ///  - At final epoch, writes out predictions.csv as "predicted value,actual value".
    pub fn train(
        &mut self,
        inputs: &Array2<f64>,   // (n, in)
        targets: &Array2<f64>,  // (n, out)
        learning_rate: f64,     // α step size
        epochs: usize,          // number of passes over the full dataset
    ) {
        // Set up a CSV file to save predictions at the very end.
        let file = File::create("predictions.csv").expect("Unable to create file");
        let mut writer = BufWriter::new(file);
        writeln!(writer, "predicted value,actual value").expect("Unable to write header");

        for epoch in 0..epochs {
            // Forward pass: compute A(L) from inputs.
            let predictions = self.forward(inputs.clone());

            // Optional: compute MSE for logging (not the actual training loss derivative).
            let loss = mean_squared_error(&predictions, targets);
            println!("Epoch {}: MSE ~ {}", epoch + 1, loss);

            // On the last epoch, export predicted and actual values row-by-row.
            if epoch + 1 == epochs {
                for (pred, target) in predictions.outer_iter().zip(targets.outer_iter()) {
                    writeln!(writer, "{},{}", pred[0], target[0]).expect("Unable to write data");
                }
            }

            // Backward pass: compute grads and apply updates to all layers.
            self.backward(targets, learning_rate);

            // Safety check: if any NaN appears in parameters, abort with a clear message.
            if self
                .layers
                .iter()
                .any(|layer| layer.weights.iter().any(|&x| x.is_nan())
                    || layer.biases.iter().any(|&x| x.is_nan()))
            {
                panic!("Encountered NaN in weights or biases during training");
            }
        }
    }

    /// Save model parameters (weights and biases) to a JSON file at `path`.
    pub fn save(&self, path: &str) {
        let file = File::create(path).expect("Unable to create file");
        serde_json::to_writer(file, &self.layers).expect("Unable to write model");
    }

    /// Load model parameters (weights and biases) from a JSON file at `path`.
    pub fn load(path: &str) -> Self {
        let file = File::open(path).expect("Unable to open file");
        let layers: Vec<DenseLayer> = serde_json::from_reader(file).expect("Unable to read model");
        NeuralNetwork { layers }
    }
}
// ------------------------------------------------------------------------------------
