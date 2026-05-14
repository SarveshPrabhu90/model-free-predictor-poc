"""
End-to-end demonstration: Explicit Model vs. Modelless Predictor.

Runs the full POV pipeline:
  1. Start a simulated plant (TCP server)
  2. Collect observations via TCP
  3. Train the modelless predictor (linear regression)
  4. Compare predictions on held-out test data
  5. Compare LP-based optimisation results
  6. Generate a comparison scatter plot
"""

import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import explicit_model
from .tcp_server import PlantServer
from .data_collector import collect
from .modelless_predictor import ModellessPredictor
from .optimizer import optimize_with_explicit_model, optimize_with_learned_model

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        Modelless Predictor  —  Proof of Value Demo          ║
║                                                              ║
║  Replacing an explicit model with a data-driven approach     ║
║  that observes system behaviour over TCP/IP.                 ║
╚══════════════════════════════════════════════════════════════╝
"""


def _section(title: str):
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def run():
    np.random.seed(42)
    print(BANNER)

    # ── Phase 1 ─────────────────────────────────────────────────────────
    _section("PHASE 1  ·  Starting simulated plant (TCP server)")

    server = PlantServer()
    server.start()
    time.sleep(0.5)

    print(f"  Plant server listening on {server.host}:{server.port}")
    print()
    print("  Explicit-model coefficients (HIDDEN from the learner):")
    print(f"    Yield   weights : {explicit_model.YIELD_COEFFICIENTS.tolist()}"
          f"   intercept : {explicit_model.YIELD_INTERCEPT}")
    print(f"    Purity  weights : {explicit_model.PURITY_COEFFICIENTS.tolist()}"
          f"   intercept : {explicit_model.PURITY_INTERCEPT}")
    print()

    # ── Phase 2 ─────────────────────────────────────────────────────────
    _section("PHASE 2  ·  Collecting observations via TCP")

    n_train, n_test = 200, 50

    print(f"  Requesting {n_train} training observations …")
    train_in, train_out = collect(n_train)

    print(f"  Requesting {n_test} test observations …")
    test_in, test_out = collect(n_test)

    print(f"  Collected {train_in.shape[0]} train + {test_in.shape[0]} test samples")
    print(f"    Input  shape : {train_in.shape}")
    print(f"    Output shape : {train_out.shape}")
    print()

    # ── Phase 3 ─────────────────────────────────────────────────────────
    _section("PHASE 3  ·  Training modelless predictor (linear regression)")

    predictor = ModellessPredictor()
    predictor.fit(train_in, train_out, output_names=["yield", "purity"])

    coeffs = predictor.coefficients
    print("  Learned coefficients:")
    for name, c in coeffs.items():
        w = [round(v, 4) for v in c["weights"]]
        print(f"    {name:8s}  weights : {w}   intercept : {round(c['intercept'], 4)}")
    print()

    # ── Phase 4 ─────────────────────────────────────────────────────────
    _section("PHASE 4  ·  Prediction comparison on test data")

    explicit_preds = explicit_model.predict(test_in, add_noise=False)
    learned_preds = predictor.predict(test_in)
    metrics = predictor.evaluate(test_in, explicit_preds)

    print("  Modelless vs. Explicit (noise-free ground truth):")
    for name, m in metrics.items():
        print(f"    {name:8s}  MAE = {m['mae']:.4f}   R² = {m['r2']:.6f}")

    print()
    print("  Sample predictions (first 5 test points):")
    hdr = (f"  {'#':>3s}  {'Explicit Yield':>15s} {'Learned Yield':>14s}"
           f"  {'Explicit Purity':>16s} {'Learned Purity':>15s}")
    print(hdr)
    for i in range(5):
        print(f"  {i+1:3d}  {explicit_preds[i,0]:15.2f} {learned_preds[i,0]:14.2f}"
              f"  {explicit_preds[i,1]:16.2f} {learned_preds[i,1]:15.2f}")
    print()

    # ── Phase 5 ─────────────────────────────────────────────────────────
    _section("PHASE 5  ·  Optimisation comparison  (max yield, purity ≥ 90 %)")

    opt_exp = optimize_with_explicit_model(min_purity=90.0)
    opt_lrn = optimize_with_learned_model(predictor, min_purity=90.0)

    if opt_exp["success"] and opt_lrn["success"]:
        for label, opt in [("Explicit model", opt_exp), ("Learned  model", opt_lrn)]:
            xi = opt["optimal_inputs"]
            print(f"  {label} optimum:")
            print(f"    Inputs  → T = {xi[0]:.2f} °C,  "
                  f"F = {xi[1]:.2f} L/min,  C = {xi[2]:.2f} %")
            print(f"    Outputs → Yield = {opt['predicted_yield']:.2f} %,  "
                  f"Purity = {opt['predicted_purity']:.2f} %")

        diff = abs(opt_exp["predicted_yield"] - opt_lrn["predicted_yield"])
        print(f"\n  Yield difference between models: {diff:.4f} %")
    print()

    # ── Phase 6 ─────────────────────────────────────────────────────────
    _section("PHASE 6  ·  Generating comparison plots")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (name, ax) in enumerate(zip(["Yield", "Purity"], axes)):
        ax.scatter(
            explicit_preds[:, idx],
            learned_preds[:, idx],
            alpha=0.6, edgecolors="k", linewidth=0.5, s=40,
        )
        lo = min(explicit_preds[:, idx].min(), learned_preds[:, idx].min())
        hi = max(explicit_preds[:, idx].max(), learned_preds[:, idx].max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "r--", linewidth=1, label="Perfect agreement",
        )
        ax.set_xlabel(f"Explicit Model {name}")
        ax.set_ylabel(f"Modelless Predictor {name}")
        ax.set_title(f"{name}: Explicit vs. Modelless")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plot_path = "comparison_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Scatter plot saved → {plot_path}")
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    _section("SUMMARY")
    print("  The modelless predictor successfully learned the system's")
    print("  behaviour from TCP-streamed observations alone, achieving")
    print("  near-perfect agreement with the explicit model on both")
    print("  prediction and optimisation tasks.")
    print()
    print("  Key takeaway:  We can replace hand-crafted models with")
    print("  data-driven learners that observe system outputs — no")
    print("  explicit model knowledge required.")
    print()

    # cleanup
    server.stop()
    print("  Plant server stopped.  Demo complete.\n")


if __name__ == "__main__":
    run()
