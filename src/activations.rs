// -------------------------------- activations.rs -----------------------------------
// This module defines activation functions used in the network.
// - ReLU for hidden layers: f(x) = max(0, x)
// - Sigmoid for output layer (binary classification): σ(x) = 1 / (1 + e^-x)

use ndarray::Array2; // Array2<f64> is a 2D matrix type from ndarray (shape: (rows, cols)).

/// Apply ReLU elementwise over a 2D array.
/// Input: (n_samples, n_units) pre- or post-activation values.
/// Output: same shape, with negatives clamped to zero.
pub fn relu(input: &Array2<f64>) -> Array2<f64> {
    // mapv applies a closure to every element, returning a new owned array.
    input.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

/// Derivative of ReLU evaluated at the *post-activation* (safe for ReLU).
/// Returns 1 where the value is positive, 0 otherwise.
pub fn relu_derivative(post_activation: &Array2<f64>) -> Array2<f64> {
    // For ReLU, sign(post_activation) == sign(pre_activation) whenever post_activation>0,
    // so we can compute derivative from post-activation values.
    post_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

/// Sigmoid activation (elementwise): σ(x) = 1 / (1 + e^-x)
/// Produces an output in (0, 1), representing probability in binary classification.
pub fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}
// ------------------------------------------------------------------------------------
