Datasheet: Black-Box Optimization Capstone Project Data Set

This datasheet follows the structured framework for documenting data sets to ensure ethical, practical use and clear communication.

1. Motivation

Purpose: This data set was created to address a Black-Box Optimization (BBO) challenge, specifically designed for an engineering and data science capstone project.

Task Support: It supports the task of finding the global maximum of eight unknown objective functions (F1–F8) with varying dimensions and characteristics (e.g., noisy vs. smooth surfaces).


2. Composition
Contents: The data set consists of input-output pairs for eight distinct functions. Inputs are coordinate arrays (ranging from 2 to 8 dimensions), and outputs are scalar values representing the objective function's response.
Size and Format: It contains 10 iterations (submissions) of data points for each of the 8 functions. The data is stored in .npy (NumPy) format 

Gaps: There is a significant exploitation bias in high-dimensional functions like F8. Due to the late-stage focus on refining a known peak at 9.9869, vast regions of the 8-dimensional search space remain entirely unexplored.

Relationships: Each instance is linked to a specific iteration of the Bayesian Optimization loop, where previous outputs informed the selection of subsequent inputs.
3. Collection Process

Query Generation: Queries were generated using a Bayesian Optimization (BO) framework.

Strategy: The process utilized a Gaussian Process Regression (GPR) surrogate model with a Matern 2.5 kernel. The acquisition function was Expected Improvement (EI), optimized via L-BFGS-B with restarts ranging from 150 to 300 to avoid local minima.

Sampling Strategy: A heterogeneous jitter (xi) mapping strategy was used to balance exploration and exploitation.

Exploration: High jitter (xi = 0.50) was used for stagnant functions (F1, F4, F6) to intercept hidden spikes.

Exploitation: Low jitter (xi = 0.001) was used for high-performing functions (F5, F8) to refine known peaks.

Time Frame: Data was collected over a 10-week period  corresponding to the submission cycle of the capstone project.

4. Preprocessing and Uses

Transformations: Input data was normalized to a unit hypercube [0, 1]. For the noisy Function 2, a variance-handling transformation was applied to the GPR model to ensure robustness against stochastic fluctuations.

Intended Uses: This data set is intended for benchmarking Bayesian Optimization algorithms and studying the performance of different acquisition function strategies in restricted-budget environments.

Inappropriate Uses: The data set is not suitable for general-purpose machine learning training outside of the BBO context, as the sampling is highly biased toward objective function peaks rather than representing a uniform distribution of the search space.

5. Distribution and Maintenance

Availability: The data set is currently maintained within a  GitHub repository and a project-specific submission portal.

Terms of Use: Use is restricted to academic review and internal project evaluation. It is not licensed for commercial use at this time.

Maintenance: The data set is not actively maintained or updated as it is for a capstone project