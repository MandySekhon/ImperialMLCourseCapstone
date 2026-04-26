Model Card: Heterogeneous Bayesian Optimization (HBO)

This model card provides documentation for the Bayesian Optimization approach used in the engineering capstone project, ensuring transparency and responsible AI governance.

1. Overview

Model Name: Heterogeneous Bayesian Optimizer (HBO).

Type: Gaussian Process Regression (GPR) based Bayesian Optimization.

Version: 10.0 

Task: Sequential global optimization of unknown objective functions (F1–F8).

2. Intended Use

Primary Tasks: Designed to find the global maximum of complex, black-box engineering functions with a limited query budget.

Target Users: Engineering researchers and data scientists specializing in simulation-based optimization.

Scenarios to Avoid: This approach is not suitable for non-smooth functions containing discrete jumps   as the GPR surrogate assumes a level of spatial correlation.

3. Details and Strategy Evolution

Core Architecture: Utilizes a Matern 2.5 kernel to balance response surface smoothness and flexibility, combined with an Expected Improvement (EI) acquisition function.

Evolution (Rounds 1–8): Initial rounds focused on broad landscape mapping and kernel hyperparameter tuning using 50 restarts.

Evolution (Rounds 9–10): The approach shifted toward high-rigor search depth, increasing optimizer restarts to 200 and eventually 300 to prevent the acquisition function from settling in sub-optimal local minima.

Dynamic Jitter Mapping: A bimodal xi (jitter) strategy was implemented to handle diverse function behaviors simultaneously:

High Exploration (xi = 0.50): Applied to stagnant functions (F1, F3, F4, F6) to break out of low-value regions.

Precision Exploitation (xi  0.001): Applied to established peaks (F5, F7, F8) to refine global maxima.

4. Performance

Metric: The primary metric is the scalar value of the objective functions across eight distinct tasks.

Optimization Results:

Function 8 (High Dimensional): Successfully reached 9.9869, approaching the 10.0 target.

Function 5 (Recovery): Demonstrated resilience by recovering from a drop (182.77) to a high-value state of 1593.32.

Function 7 (High Precision): Achieved a stable peak of 2.55.

Stagnant Zones: Functions 1, 3, 4, and 6 remained in negative or near-zero territory (e.g., F4 at -25.27), highlighting the difficulty of high-dimensional discovery.

5. Assumptions and Limitations

Smoothness Assumption: The model assumes that the underlying functions are twice-differentiable and follow a Matern 2.5 spatial distribution; violation of this leads to "oversmoothing" and missed peaks.

Exploitation Bias: The strategy is biased toward refining known high-value areas, which leaves large sections of the high-dimensional search space (e.g., 8D for F8) entirely unexplored.

Computational Cost: Increasing restarts to 300 significantly increases the time required per query, which may be a constraint in real-time or low-latency applications.

6. Ethical Considerations

Transparency and Reproducibility: By documenting the specific xi maps and kernel hyperparameters, the logic behind every query is auditable and reproducible by external researchers.

Bias Awareness: Identifying the exploitation bias ensures that stakeholders understand the model is not claiming to have found the absolute global maximum, but rather the best local maximum within the sampled regions.