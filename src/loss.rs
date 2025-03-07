use ndarray::Array2;

/// Computes the Mean Squared Error (MSE) between predictions and targets.
pub fn mean_squared_error(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let diff = predictions - targets;
    diff.mapv(|x| x.powi(2)).mean().unwrap()
}
