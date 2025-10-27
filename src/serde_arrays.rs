// ------------------------------- serde_arrays.rs -----------------------------------
// Custom Serde (de)serializers for ndarray::Array2<f64>.
// We store arrays as a tuple: (shape: Vec<usize>, data: Vec<f64>).
// On load, we reconstruct an Array2 from (rows, cols) and the flattened data.

use ndarray::Array2;                         // 2D array type to (de)serialize.
use serde::{Deserialize, Deserializer, Serialize, Serializer}; // Serde traits/macros.

/// Serialize Array2<f64> by extracting shape and flattened data.
pub fn serialize<S>(array: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let shape = array.shape().to_vec();                         // e.g., vec![rows, cols]
    let data = array.iter().cloned().collect::<Vec<f64>>();     // Flatten row-major data.
    (&shape, &data).serialize(serializer)                       // Serialize as tuple for compactness.
}

/// Deserialize Array2<f64> from (shape, data).
pub fn deserialize<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    // Expect (Vec<usize>, Vec<f64>) format during JSON decode.
    let (shape, data): (Vec<usize>, Vec<f64>) = Deserialize::deserialize(deserializer)?;

    // Validate we indeed got a 2D shape like [rows, cols].
    if shape.len() != 2 {
        return Err(serde::de::Error::custom("Expected 2D shape"));
    }

    // Rebuild Array2 with provided shape; map any size mismatch into a serde error.
    Array2::from_shape_vec((shape[0], shape[1]), data).map_err(serde::de::Error::custom)
}
// ------------------------------------------------------------------------------------
