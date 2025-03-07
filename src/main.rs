use clap::{Parser, Subcommand}; // Import clap to create and parse command-line interfaces.
use rust_net::{data::load_csv, train::NeuralNetwork}; // Import necessary modules for data loading and neural network operations.
use ndarray::Array2; // Import Array2 to handle 2D arrays used in the neural network computations.

/// The main Command Line Interface (CLI) structure for the RustNet application.
/// This structure defines the top-level interface that the user interacts with.
/// Users can either train the model or make predictions using commands provided through this interface.
#[derive(Parser)]
#[command(name = "RustNetCLI")] // Defines the program's name as RustNetCLI.
#[command(about = "A neural network library for RF wave detection", long_about = None)] // Describes the purpose of this program.
struct CLI {
    #[command(subcommand)] // Specifies that the CLI has subcommands (Train and Predict).
    command: Commands, // Subcommands available to the user.
}

/// Enum to define the possible subcommands for the CLI.
/// Subcommands allow users to specify whether they want to train the model or make predictions.
#[derive(Subcommand)]
enum Commands {
    /// Subcommand to train the neural network using a dataset.
    Train {
        /// The path to the dataset file (in CSV format) provided by the user.
        #[arg(short, long)] // Allows the argument to be passed with `-d` or `--data`.
        data: String,
    },
    /// Subcommand to use a pre-trained model for making predictions.
    Predict {
        /// The path to the saved model file (in JSON format).
        #[arg(short, long)] // Allows the argument to be passed with `-m` or `--model`.
        model: String,
        /// The path to the input dataset file (in CSV format) for predictions.
        #[arg(short, long)] // Allows the argument to be passed with `-i` or `--input`.
        input: String,
    },
}

/// The main function is the entry point of the program.
/// It parses the command-line arguments, processes the user's command, and performs the requested action.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting RustNetCLI program..."); // Indicate that the program has started.

    // Parse the command-line arguments provided by the user.
    // The parsed arguments are stored in the `cli` variable.
    let cli = CLI::parse();

    // Match the parsed command to determine whether the user wants to train or predict.
    match cli.command {
        Commands::Train { data } => {
            println!("Training mode activated..."); // Notify the user that the training mode has been activated.

            // Step 1: Load the dataset from the specified CSV file.
            let dataset = load_csv(&data)?; // `load_csv` reads the file and converts it into a 2D ndarray.
            println!("Loaded dataset from '{}'", data); // Confirm successful dataset loading.

            // Step 2: Split the dataset into input features (`inputs`) and target labels (`targets`).
            let split_index = dataset.ncols() - 1; // Use the last column of the dataset as the target labels.
            let (inputs, targets) = dataset.view().split_at(ndarray::Axis(1), split_index);

            // Step 3: Convert the slices into owned arrays.
            // The neural network requires owned arrays for processing, as slices are immutable views.
            let inputs = inputs.to_owned();
            let targets = targets.to_owned();

            // Step 4: Normalize the input features.
            // Normalization ensures that each feature has a mean of 0 and a standard deviation of 1.
            // This helps the neural network train more effectively by avoiding issues with scale differences.
            let inputs = normalize(&inputs);
            println!(
                "Input shape: {:?}, Target shape: {:?}",
                inputs.dim(),  // Dimensions of the input array (e.g., 1000 samples, 2 features).
                targets.dim()  // Dimensions of the target array (e.g., 1000 samples, 1 target).
            );

            // Step 5: Initialize the neural network with the specified architecture.
            // The architecture consists of:
            // - An input layer with a size equal to the number of input features.
            // - A hidden layer with 10 neurons.
            // - An output layer with a size equal to the number of target columns.
            let mut nn = NeuralNetwork::new(vec![inputs.ncols(), 10, targets.ncols()]);
            println!(
                "Created neural network with layers: {:?}",
                vec![inputs.ncols(), 10, targets.ncols()] // Display the network architecture.
            );

            // Step 6: Train the neural network.
            // The training process involves:
            // - Forward propagation: Compute predictions for the inputs.
            // - Backpropagation: Compute gradients and update the weights and biases.
            // - Loss computation: Measure how far the predictions are from the targets.
            // Parameters:
            // - Learning rate: 0.001 (controls the step size for weight updates).
            // - Epochs: 1000 (number of times the entire dataset is passed through the network).
            nn.train(&inputs, &targets, 0.001, 1000);
            println!("Training complete."); // Notify the user that training is complete.

            // Step 7: Save the trained model to a JSON file.
            // The model can be loaded later for making predictions on new data.
            nn.save("model.json");
            println!("Model saved as 'model.json'"); // Confirm the model has been saved.
        }
        Commands::Predict { model, input } => {
            println!("Prediction mode activated..."); // Notify the user that the prediction mode has been activated.

            // Step 1: Load the pre-trained model from the specified JSON file.
            // The model file contains the network's weights and biases.
            let mut nn = NeuralNetwork::load(&model);
            println!("Loaded model from '{}'", model); // Confirm successful model loading.

            // Step 2: Load the input dataset for making predictions.
            let dataset = load_csv(&input)?; // Read the input data from the specified CSV file.
            let split_index = dataset.ncols() - 1; // Split the dataset, though targets are ignored in this step.
            let (inputs, _) = dataset.view().split_at(ndarray::Axis(1), split_index);

            // Step 3: Convert the input slices into an owned array and normalize it.
            let inputs = inputs.to_owned(); // Convert the slice into an owned array.
            let inputs = normalize(&inputs); // Normalize the input features.
            println!(
                "Loaded input data from '{}', shape: {:?}",
                input,
                inputs.dim() // Display the dimensions of the input array.
            );

            // Step 4: Perform predictions using the loaded model.
            // The model processes the inputs through its layers and outputs predictions.
            let predictions = nn.forward(inputs);
            println!("Predictions:"); // Notify the user that predictions are being displayed.

            // Print each prediction row for review.
            for row in predictions.outer_iter() {
                println!("{:?}", row); // Display the predictions in a readable format.
            }
        }
    }

    Ok(()) // Return success status.
}

/// Helper function to normalize the input features.
/// - This function adjusts each feature to have a mean of 0 and a standard deviation of 1.
/// - Normalization helps the neural network converge faster and prevents large differences in feature scales from dominating the training process.
/// Parameters:
/// - `inputs`: A 2D ndarray of input features.
/// Returns: A 2D ndarray of normalized features.
fn normalize(inputs: &Array2<f64>) -> Array2<f64> {
    let mean = inputs.mean_axis(ndarray::Axis(0)).unwrap(); // Compute the mean of each feature.
    let std = inputs.std_axis(ndarray::Axis(0), 0.0);      // Compute the standard deviation of each feature.
    (inputs - &mean) / &std // Normalize each feature: (value - mean) / std deviation.
}
