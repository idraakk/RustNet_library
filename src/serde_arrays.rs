use ndarray::Array2;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serializes an Array2 by storing its shape and flattened data.
pub fn serialize<S>(array: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let shape = array.shape().to_vec();
    let data = array.iter().cloned().collect::<Vec<f64>>();
    (&shape, &data).serialize(serializer)
}

/// Deserializes an Array2 from its shape and flattened data.
pub fn deserialize<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let (shape, data): (Vec<usize>, Vec<f64>) = Deserialize::deserialize(deserializer)?;
    Array2::from_shape_vec((shape[0], shape[1]), data).map_err(serde::de::Error::custom)
}
