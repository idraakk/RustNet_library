use ndarray::Array2;

/// Applies the ReLU activation elementwise.
pub fn relu(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

/// Computes the derivative of ReLU.
pub fn relu_derivative(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

/// Applies the Sigmoid activation elementwise.
pub fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
    input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Computes the derivative of the Sigmoid function.
pub fn sigmoid_derivative(input: &Array2<f64>) -> Array2<f64> {
    let sig = sigmoid(input);
    &sig * &(1.0 - &sig)
}
