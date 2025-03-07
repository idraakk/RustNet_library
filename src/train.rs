use crate::activations::*;
use crate::layers::*;
use crate::loss::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Write, BufWriter};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    /// Initializes the neural network given a list of layer sizes.
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i+1]));
        }
        NeuralNetwork { layers }
    }

    /// Performs a forward pass through all layers.
    pub fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            input = layer.forward(&input);
            if i < num_layers - 1 {
                input = relu(&input);
            }
            layer.outputs = Some(input.clone());
        }
        input
    }

    /// Backpropagation to update weights and biases.
    pub fn backward(&mut self, targets: &Array2<f64>, learning_rate: f64) {
        let num_layers = self.layers.len();
        let mut gradient = None;

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let outputs = layer.outputs.as_ref().unwrap();
            let inputs = layer.inputs.as_ref().unwrap();
            if i == num_layers - 1 {
                let d_z = outputs - targets;
                let d_w = inputs.t().dot(&d_z) / inputs.dim().0 as f64;
                let db = d_z.sum_axis(ndarray::Axis(0)) / inputs.dim().0 as f64;
                layer.weights -= &(d_w * learning_rate);
                layer.biases -= &(db.insert_axis(ndarray::Axis(0)) * learning_rate);
                gradient = Some(d_z.dot(&layer.weights.t()));
            } else {
                let d_a = gradient.unwrap();
                let d_z = d_a * relu_derivative(outputs);
                let d_w = inputs.t().dot(&d_z) / inputs.dim().0 as f64;
                let db = d_z.sum_axis(ndarray::Axis(0)) / inputs.dim().0 as f64;
                layer.weights -= &(d_w * learning_rate);
                layer.biases -= &(db.insert_axis(ndarray::Axis(0)) * learning_rate);
                gradient = Some(d_z.dot(&layer.weights.t()));
            }
        }
    }

    /// Trains the neural network.
    /// At the final epoch, writes a "predictions.csv" file with header and predicted/actual values.
    pub fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, learning_rate: f64, epochs: usize) {
        let file = File::create("predictions.csv").expect("Unable to create file");
        let mut writer = BufWriter::new(file);
        writeln!(writer, "predicted value,actual value").expect("Unable to write header");

        for epoch in 0..epochs {
            let predictions = self.forward(inputs.clone());
            let loss = mean_squared_error(&predictions, targets);
            println!("Epoch {}: Loss = {}", epoch, loss);

            if epoch == epochs - 1 {
                for (pred, target) in predictions.outer_iter().zip(targets.outer_iter()) {
                    // Assumes single-output (one value per row)
                    writeln!(writer, "{},{}", pred[0], target[0]).expect("Unable to write data");
                }
            }

            self.backward(targets, learning_rate);

            if self.layers.iter().any(|layer| layer.weights.iter().any(|&x| x.is_nan()) ||
                                               layer.biases.iter().any(|&x| x.is_nan())) {
                panic!("Encountered NaN in weights or biases during training");
            }
        }
    }

    /// Saves the trained model to a JSON file.
    pub fn save(&self, path: &str) {
        let file = File::create(path).expect("Unable to create file");
        serde_json::to_writer(file, &self.layers).expect("Unable to write model");
    }

    /// Loads a model from a JSON file.
    pub fn load(path: &str) -> Self {
        let file = File::open(path).expect("Unable to open file");
        let layers: Vec<DenseLayer> = serde_json::from_reader(file).expect("Unable to read model");
        NeuralNetwork { layers }
    }
}
