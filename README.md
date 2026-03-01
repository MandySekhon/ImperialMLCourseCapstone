# ImperialMLCourseCapstone
Section 1: project overview

This project focuses on Sequential Model-Based Optimization of "black-box" functions, mathematical systems where the internal formula is unknown, and the only way to learn is through expensive trial-and-error queries.

The Goal: To find the global maximum of various hidden functions using a strictly limited budget

Real-World Relevance: This mirrors high-stakes ML scenarios such as Hyperparameter Tuning , drug discovery, or engineering hardware where each "test" is costly, time-consuming, or destructive.

Career Impact: This capstone develops probabilistic thinking. It shifts the focus from Big Data processing to Small Data efficiency, teaching me how to make optimal decisions under extreme uncertainty.

Section 2: Inputs and Outputs

The model interacts with the black-box functions through a structured query system.

Inputs: * An n-dimensional vector (ranging from 2D to 8D).

Each dimension is constrained to the range [0, 1]

Outputs:  A single float value representing the function's response
Section 3: Challenge Objectives
The core objective is to maximize the response value across eight distinct functions.

Constraints & Limitations:
Query Budget: Only a limited number of total queries are allowed across all functions. Efficiency is paramount.

Unknown Structure: Each function has different properties (some are smooth, some are noisy, some have irrelevant dimensions).

Optimization Difficulty: High-dimensional functions (up to 8D) create a "curse of dimensionality," making it easy to miss narrow global peaks.

Section 4: Technical Approach
My strategy has evolved from broad discovery to high-precision refinement over the course of three submissions.

Machine Learning Methods
I utilize Gaussian Process Regression (GPR) to build a surrogate model of the unknown functions.

Kernel Choice: I use the Matern 5/2 kernel, which offers a realistic balance of smoothness and flexibility compared to a standard RBF kernel.

Hyperparameter Tuning: Instead of fixed heuristics, I use Automatic Relevance Determination (ARD). By increasing to 30 GPR restarts, the model independently learns the length scale for each dimension, identifying which inputs are active and which are irrelevant.

Classification vs. Regression: While a Soft-Margin SVM could identify high-performance zones, I chose GPR because it provides a probabilistic uncertainty which is vital for the acquisition function.

Exploration vs. Exploitation
I balance these competing needs using the Expected Improvement (EI) acquisition function:

Exploration: In early rounds, a higher jitter (xi = 0.01) was used to probe unknown areas.

Exploitation: For Submission 3, I reduced jitter to 0.005 to climb the peaks discovered in previous rounds (notably Function 5).

Search Rigor: To ensure the global peak is found on the surrogate map, I perform 40 restarts of the acquisition optimizer for every query.
