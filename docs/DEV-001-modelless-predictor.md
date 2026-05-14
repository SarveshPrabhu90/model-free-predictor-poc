# DEV-001: Modelless Predictor — Proof of Value Implementation

---

## Objective

Demonstrate that an intelligent, **model-free** solution can make predictions
equivalent to an explicit mathematical model by observing a small number of
the system's input-output pairs — without any knowledge of the underlying
model coefficients.

## Background

The current system relies on a hand-crafted explicit model with known
coefficients to predict process outputs (yield, purity) from control inputs
(temperature, flow rate, catalyst concentration). Maintaining and updating
this model is costly and error-prone. A data-driven replacement that learns
from observations would be more adaptable and easier to deploy.

## Acceptance Criteria

- [x] Simulated plant with a hidden explicit model serving as ground truth
- [x] TCP/IP server streaming JSON-encoded input-output observations
- [x] TCP client collecting training and test data sets
- [x] Modelless predictor using linear regression to learn from observations
- [x] Prediction accuracy: R² ≥ 0.999 against noise-free ground truth
- [x] LP optimizer finding optimal inputs with both explicit and learned models
- [x] Optimization results match within < 1 % yield difference
- [x] Comparison scatter plot generated automatically
- [x] End-to-end demo runnable with a single command (`python run_demo.py`)

## Technical Approach

### Architecture

```
Simulated Plant ──TCP/IP──► Data Collector ──► Modelless Predictor ──► LP Optimizer
(explicit_model)            (data_collector)    (linear regression)     (scipy.linprog)
(tcp_server)
```

### Key Technologies

| Area            | Tool / Library              | Purpose                                 |
|-----------------|-----------------------------|-----------------------------------------|
| Language        | Python 3.10+               | All implementation                      |
| Networking      | `socket` (stdlib)          | TCP server/client for data streaming    |
| Learning        | scikit-learn `LinearRegression` | Fit input-output relationships     |
| Optimization    | scipy `linprog` (HiGHS)   | Maximize yield subject to purity constraint |
| Visualization   | matplotlib                 | Explicit vs. learned scatter plots      |

### Modules

| File                     | Responsibility                                   |
|--------------------------|--------------------------------------------------|
| `src/explicit_model.py`  | Ground-truth model with hidden coefficients       |
| `src/tcp_server.py`      | TCP server exposing plant observations            |
| `src/data_collector.py`  | TCP client gathering input-output pairs           |
| `src/modelless_predictor.py` | Linear regression learner                     |
| `src/optimizer.py`       | LP optimization using explicit and learned models |
| `src/demo.py`            | End-to-end 6-phase demo orchestration             |
| `run_demo.py`            | Convenience entry point                           |

## Results

| Metric              | Value          |
|----------------------|---------------|
| Training samples     | 200           |
| Test samples         | 50            |
| Yield MAE            | 0.0527        |
| Yield R²             | 0.9999        |
| Purity MAE           | 0.0565        |
| Purity R²            | 0.9996        |
| Optimization match   | < 0.01 % diff |

## How to Run

```bash
cd modelless-predictor
pip install -r requirements.txt
python run_demo.py
```

## Future Work

- Replace linear regression with online/incremental learning for real-time adaptation
- Swap the simulated TCP server for a real OPC-UA or MQTT data source
- Add non-linear learners (e.g., ridge regression, neural nets) for more complex plants
- Integrate with a dashboard for live monitoring of prediction drift
