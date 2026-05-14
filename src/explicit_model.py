"""
Explicit Model — The 'known' model we want to replace.

Simulates a pharmaceutical process (e.g., a mixing/reaction vessel).
This represents the current state: a hand-crafted mathematical model with
known coefficients used to predict process outputs from control inputs.

Inputs (control variables):
    temperature      (°C)   : 20 – 80
    flow_rate        (L/min): 1 – 10
    concentration    (%)    : 0.1 – 5.0

Outputs (process responses):
    yield   (%)  — product yield
    purity  (%)  — product purity
"""

import numpy as np

# ── Ground-truth coefficients (unknown to the modelless predictor) ──────────

YIELD_COEFFICIENTS = np.array([0.45, 0.30, 0.80])
YIELD_INTERCEPT = 12.0

PURITY_COEFFICIENTS = np.array([-0.15, 0.55, 0.35])
PURITY_INTERCEPT = 85.0

INPUT_RANGES = {
    "temperature": (20.0, 80.0),
    "flow_rate": (1.0, 10.0),
    "concentration": (0.1, 5.0),
}

NOISE_STD = 0.5  # Simulated process noise


# ── Public helpers ──────────────────────────────────────────────────────────

def predict(inputs: np.ndarray, add_noise: bool = True) -> np.ndarray:
    """
    Compute outputs using the explicit model.

    Args:
        inputs:    (N, 3) array of [temperature, flow_rate, concentration]
        add_noise: whether to add simulated process noise

    Returns:
        (N, 2) array of [yield, purity]
    """
    inputs = np.atleast_2d(inputs)

    y = inputs @ YIELD_COEFFICIENTS + YIELD_INTERCEPT
    p = inputs @ PURITY_COEFFICIENTS + PURITY_INTERCEPT

    if add_noise:
        y = y + np.random.normal(0, NOISE_STD, size=y.shape)
        p = p + np.random.normal(0, NOISE_STD, size=p.shape)

    return np.column_stack([y, p])


def random_inputs(n: int) -> np.ndarray:
    """Generate *n* random input vectors within valid operating ranges."""
    rng = np.random.default_rng()
    cols = [rng.uniform(lo, hi, size=n) for lo, hi in INPUT_RANGES.values()]
    return np.column_stack(cols)
