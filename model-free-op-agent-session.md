# Coding Agent Session: Model-Free Optimizer POC

Goal: Build a public, non-confidential Python prototype for testing model-free optimization on noisy physical calibration problems.

What I asked the agent to help with:
- Set up the project structure
- Implement a simulated device response
- Add optimizer experiments and baselines
- Add metrics like MAE, RMSE, and max absolute error
- Generate reproducible logs
- Refactor the code so the prototype was easier to explain and extend

What I drove:
- Framed the physical calibration problem
- Defined the objective/loss functions
- Chose which optimizers were worth testing
- Interpreted whether the results made sense for a real hardware-in-the-loop system
- Kept the project non-confidential and separate from company code

This session helped turn an ambiguous physical control problem into a runnable prototype, and needed me to define the system behavior, evaluation criteria, constraints, and what “good” meant.
