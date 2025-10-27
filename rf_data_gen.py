# -------------------------------- rf_data_gen.py -----------------------------------
# Purpose:
#   Generate a simple, synthetic RF-like dataset and save it to 'rf_data.csv'.
#   Each row contains:
#     - Frequency (continuous feature)
#     - Amplitude (continuous feature): sine wave with Gaussian noise
#     - Label (binary target): 1 if amplitude > 0 else 0
#
# Why this design?
#   - A noisy sine wave is a classic, smooth, periodic signal that loosely mimics
#     a simple RF pattern. Classifying sign(amplitude) is an approachable binary
#     task for testing the library’s ability to learn non-linear boundaries.
#
# How to use:
#   1) Ensure your Python environment has NumPy and Pandas installed.
#      (In your venv: `pip install numpy pandas`)
#   2) Run this script: `python rf_data_gen.py`
#   3) A file named 'rf_data.csv' will be created in the current directory.
#   4) Use run2.py to train/evaluate the Rust network on this dataset.

import numpy as np      # Numerical computing: arrays, random numbers, vectorized math
import pandas as pd     # Tabular data structures (DataFrame) and convenient CSV I/O
from pathlib import Path  # Cross-platform filesystem paths (optional, used for friendly messages)

def main() -> None:
    """
    Generate and persist a synthetic dataset that looks like (noisy) RF samples.

    Configuration choices below are intentionally simple and explicit so you can
    modify them easily if you want to change sample size, noise level, or the
    sine function’s interpretation (radians vs cycles).
    """

    # ----------------------------- Reproducibility -----------------------------
    # Using NumPy's Generator API (preferred) with a fixed seed ensures that each
    # run produces exactly the same random noise — critical for debugging and fair
    # comparisons across model changes.
    rng = np.random.default_rng(seed=42)

    # --------------------------- Sample configuration --------------------------
    # Number of points in our synthetic time/frequency axis.
    # - Larger values produce more samples (more training data).
    # - Smaller values speed up generation and training.
    num_samples: int = 1000

    # "Frequency" axis definition:
    # We construct a linearly spaced array from 0 to 100 with 'num_samples' points.
    # IMPORTANT: Here, we interpret these values as *radians* when we pass them to np.sin().
    # If you want to interpret 'frequency' as "cycles", use sin(2π * cycles) instead —
    # see the 'amplitude' block below for a commented alternative.
    frequency_start: float = 0.0
    frequency_end: float = 100.0
    frequency: np.ndarray = np.linspace(frequency_start, frequency_end, num_samples)

    # --------------------------- Signal construction ---------------------------
    # Amplitude is a sine wave + Gaussian noise.
    # - Base signal: np.sin(frequency) where 'frequency' are radians.
    # - Noise: rng.normal(mean=0, std_dev, size)
    # - You can adjust 'noise_std' to make classification easier/harder:
    #     * Lower noise_std -> cleaner boundary; easier to classify
    #     * Higher noise_std -> blurrier boundary; harder to classify
    noise_mean: float = 0.0
    noise_std: float = 0.1

    # Base sine wave (radians interpretation):
    base_signal: np.ndarray = np.sin(frequency)

    # Alternative (cycles interpretation):
    #   cycles = np.linspace(0.0, 10.0, num_samples)      # 10 cycles across the domain
    #   base_signal = np.sin(2.0 * np.pi * cycles)        # Convert cycles -> radians
    # To switch to cycles: comment out the radians version above, uncomment the two
    # lines here, and also consider storing 'cycles' as the first feature instead
    # of 'frequency' (purely stylistic—either way is fine for a toy dataset).

    # Random Gaussian noise:
    noise: np.ndarray = rng.normal(loc=noise_mean, scale=noise_std, size=num_samples)

    # Final amplitude: base signal + noise
    amplitude: np.ndarray = base_signal + noise

    # ------------------------------ Label creation -----------------------------
    # Binary label — 1 when amplitude > 0 (above zero line), else 0.
    # This turns our continuous signal into a classification problem.
    labels: np.ndarray = (amplitude > 0.0).astype(int)

    # ---------------------------- Data frame assembly --------------------------
    # We package the columns into a tidy table with explicit names:
    # - 'Frequency': the x-axis / phase-like variable we generated above
    # - 'Amplitude': the noisy sine value at that "frequency"
    # - 'Label': the binary class dependent on sign of 'Amplitude'
    df: pd.DataFrame = pd.DataFrame(
        {
            "Frequency": frequency,   # Continuous feature #1
            "Amplitude": amplitude,   # Continuous feature #2
            "Label": labels,          # Binary target in {0, 1}
        }
    )

    # ------------------------------- Persist CSV -------------------------------
    # Save to CSV with no DataFrame index column (cleaner for ML ingestion).
    out_path = Path("rf_data.csv")
    df.to_csv(out_path, index=False)

    # Friendly confirmation message:
    print(f"Dataset saved as {out_path.resolve()}")

# Standard Python entrypoint pattern — only run main() when executed as a script.
if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------------
