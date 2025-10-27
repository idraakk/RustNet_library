// ----------------------------------- loss.rs ---------------------------------------
// Loss functions for diagnostics. We log Mean Squared Error (MSE) even though
// the training uses a BCE-style gradient (dZ = A - Y) at the sigmoid output layer.

use ndarray::Array2; // 2D arrays for predictions and targets.

/// Mean Squared Error: mean((y_hat - y)^2).
/// We use it for logging to get a rough sense of convergence.
pub fn mean_squared_error(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let diff = predictions - targets;                   // Elementwise residuals.
    diff.mapv(|x| x.powi(2)).mean().unwrap_or(0.0)     // Square, average over all elements.
}
// ------------------------------------------------------------------------------------
