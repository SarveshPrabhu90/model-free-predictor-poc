# Modelless Predictor — Proof of Value

**Demonstrating that a data-driven approach can replace an explicit model by observing system behavior.**

## Core Idea

We have a simulated pharmaceutical process (e.g., a mixing/reaction vessel) that
currently uses an **explicit mathematical model** with known coefficients to predict
outputs like *yield* and *purity* from control inputs like *temperature*, *flow rate*,
and *catalyst concentration*.

This project proves it is **feasible to replace that explicit model** with an
intelligent, **modelless** solution that:

1. **Observes** a handful of input-output pairs streamed over TCP/IP
2. **Learns** the system's behavior via linear regression
3. **Predicts** future outputs with near-identical accuracy
4. **Optimizes** control inputs using linear programming — matching the
   explicit model's optimal solution

## Architecture

```
┌──────────────────────┐         TCP/IP          ┌──────────────────────┐
│   Simulated Plant    │ ◄──────────────────────► │   Data Collector     │
│  (Explicit Model)    │   JSON over sockets      │   (TCP Client)       │
│  tcp_server.py       │                          │   data_collector.py  │
└──────────────────────┘                          └─────────┬────────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │  Modelless Predictor │
                                                  │  (Linear Regression) │
                                                  │  modelless_predictor │
                                                  └─────────┬────────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────────┐
                                                  │     LP Optimizer     │
                                                  │  (scipy.linprog)     │
                                                  │  optimizer.py        │
                                                  └──────────────────────┘
```

## Key Technologies

| Area                  | Tool / Technique             |
|-----------------------|------------------------------|
| Language              | Python 3.10+                 |
| Networking            | TCP/IP sockets, JSON protocol|
| Learning              | Linear Regression (sklearn)  |
| Optimization          | Linear Programming (scipy)   |
| Visualization         | Matplotlib                   |

## Quick Start

```bash
cd modelless-predictor
pip install -r requirements.txt
python run_demo.py
```

## What the Demo Shows

1. **Phase 1** — Starts a TCP server simulating the real process plant
2. **Phase 2** — Collects 200 training + 50 test observations over TCP
3. **Phase 3** — Trains the modelless predictor (no model knowledge used)
4. **Phase 4** — Compares predictions: learned vs. explicit (R², MAE)
5. **Phase 5** — Compares optimization: finds optimal inputs with both models
6. **Phase 6** — Generates a scatter plot (`comparison_plot.png`)

## Next Steps / Code Organization

- ~~Move the model into a separate project and workspace, commit to GitHub as a separate repo. Create a ticket in the new repo to commit the PR and push it to mainline. Create unit tests.~~ **Done** → [explicit-model](https://github.com/SarveshPrabhu90/explicit-model)
- ~~Move the modelless predictor into a separate project and workspace, commit to GitHub as a separate repo. Create a ticket in the new repo to commit the PR and push it to mainline. Create unit tests.~~ **Done** → [modelless-predictor](https://github.com/SarveshPrabhu90/modelless-predictor)
- ~~Convert the current project into a web site for reporting, analysis and documentation. Considering Flask app, but looking for suggestions. Create a ticket in the new repo to commit the PR and push it to mainline.~~ **Done** → [predictor-dashboard](https://github.com/SarveshPrabhu90/predictor-dashboard)

## Project Structure

```
modelless-predictor/
├── README.md
├── requirements.txt
├── run_demo.py                  # Entry point
└── src/
    ├── __init__.py
    ├── explicit_model.py        # Ground-truth model (what we're replacing)
    ├── tcp_server.py            # TCP server exposing plant observations
    ├── data_collector.py        # TCP client gathering observations
    ├── modelless_predictor.py   # Linear regression learner
    ├── optimizer.py             # LP optimization with both models
    └── demo.py                  # End-to-end orchestration
```
