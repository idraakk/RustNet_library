// ---------------------------------- data.rs ----------------------------------------
// This helper provides a convenience function to read numeric CSVs into an ndarray::Array2<f64>.
// It assumes all fields are numeric and that all rows have equal length.

use csv::Reader;          // CSV reader from the 'csv' crate.
use ndarray::Array2;      // 2D array type for storing table-like data.
use std::error::Error;    // Trait object for returning any error type (boxed dyn Error).

/// Load a CSV file into a 2D f64 array (rows = samples, cols = features/targets).
/// Panics if a field is non-numeric. Returns a boxed error for IO/formatting issues.
pub fn load_csv(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    // Create a CSV reader from a file path. Errors propagate via '?'.
    let mut reader = Reader::from_path(file_path)?;

    // We'll collect each parsed row as Vec<f64> into a Vec<Vec<f64>>.
    let mut records = Vec::new();

    // Iterate over the CSV records. Each 'result' may be an IO error or a parsed record.
    for result in reader.records() {
        let record = result?; // '?' unwraps or returns early with the error.
        // Parse each field in the row as f64. If a field is non-numeric, panic with a clear message.
        let row: Vec<f64> = record
            .iter()
            .map(|field| field.parse::<f64>().expect("Non-numeric value in CSV"))
            .collect();
        records.push(row);
    }

    // If the CSV was empty, rows = 0 and cols = 0. Otherwise get the length from the first row.
    let rows = records.len();
    let cols = records.first().map(|r| r.len()).unwrap_or(0);

    // Flatten Vec<Vec<f64>> into a single Vec<f64> in row-major order for ndarray constructor.
    let flattened: Vec<f64> = records.into_iter().flatten().collect();

    // Construct a 2D array with shape (rows, cols) from the flat data. Error if size mismatches.
    Ok(Array2::from_shape_vec((rows, cols), flattened)?)
}
// ------------------------------------------------------------------------------------
