use csv::Reader;
use ndarray::Array2;
use std::error::Error;

/// Loads a CSV file into a 2D ndarray.
/// Each row corresponds to a data sample.
pub fn load_csv(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut records = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|field| field.parse::<f64>().unwrap()).collect();
        records.push(row);
    }

    let rows = records.len();
    let cols = records[0].len();
    let flattened: Vec<f64> = records.into_iter().flatten().collect();

    Ok(Array2::from_shape_vec((rows, cols), flattened)?)
}
