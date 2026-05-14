"""
Optimizer — Finds optimal control inputs via Linear Programming.

Demonstrates that the learned (modelless) model can drive the *same*
optimisation formulation as the explicit model, yielding equivalent results.

Problem:  maximise **yield** subject to **purity ≥ threshold**
          within the valid operating ranges for each input variable.
"""

import numpy as np
from scipy.optimize import linprog

from . import explicit_model
from .modelless_predictor import ModellessPredictor


def _bounds() -> list[tuple[float, float]]:
    """Operating bounds for each input variable."""
    return list(explicit_model.INPUT_RANGES.values())


# ── optimise with the explicit (known) model ───────────────────────────────

def optimize_with_explicit_model(min_purity: float = 90.0) -> dict:
    """
    Maximise yield using the explicit model's known coefficients.

    yield  = Yw · x + Yi   →  minimise  -Yw · x
    purity = Pw · x + Pi   →  constraint  Pw · x ≥ (min_purity - Pi)
    """
    c = -explicit_model.YIELD_COEFFICIENTS  # negate for minimisation

    A_ub = [-explicit_model.PURITY_COEFFICIENTS]
    b_ub = [-(min_purity - explicit_model.PURITY_INTERCEPT)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=_bounds(), method="highs")

    if result.success:
        x = result.x
        outputs = explicit_model.predict(x.reshape(1, -1), add_noise=False)
        return {
            "success": True,
            "optimal_inputs": x.tolist(),
            "predicted_yield": float(outputs[0, 0]),
            "predicted_purity": float(outputs[0, 1]),
            "method": "explicit_model",
        }
    return {"success": False, "message": result.message, "method": "explicit_model"}


# ── optimise with the learned (modelless) model ────────────────────────────

def optimize_with_learned_model(
    predictor: ModellessPredictor,
    min_purity: float = 90.0,
) -> dict:
    """
    Maximise yield using coefficients the predictor learned from data.

    Same LP formulation — only the coefficient source differs.
    """
    coeffs = predictor.coefficients
    yield_w = np.array(coeffs["yield"]["weights"])
    purity_w = np.array(coeffs["purity"]["weights"])
    purity_i = coeffs["purity"]["intercept"]

    c = -yield_w
    A_ub = [-purity_w]
    b_ub = [-(min_purity - purity_i)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=_bounds(), method="highs")

    if result.success:
        x = result.x
        predicted = predictor.predict(x.reshape(1, -1))
        return {
            "success": True,
            "optimal_inputs": x.tolist(),
            "predicted_yield": float(predicted[0, 0]),
            "predicted_purity": float(predicted[0, 1]),
            "method": "learned_model",
        }
    return {"success": False, "message": result.message, "method": "learned_model"}
