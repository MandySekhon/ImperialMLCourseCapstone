# Imperial ML Course Capstone: Sequential Model-Based Optimization

## Section 1: Project Overview
This project focuses on the Sequential Model-Based Optimization (SMBO) of "black-box" functions—mathematical systems where the underlying formula is unknown and the only way to learn is through expensive trial-and-error queries. 

* **The Goal:** To identify the global maximum of eight hidden functions within a strictly limited budget of total queries.
* **Real-World Relevance:** This mirrors high-stakes machine learning scenarios such as hyperparameter tuning, drug discovery, or engineering hardware where each "test" is costly or time-consuming.
* **Career Impact:** This capstone shifts the focus from "Big Data" processing to **"Small Data" efficiency**, emphasizing optimal decision-making under extreme uncertainty.

## Section 2: Inputs and Outputs
The model interacts with the black-box functions through a structured query system:
* **Inputs:** An $n$-dimensional vector (ranging from 2D to 8D) where each dimension is constrained to the range $[0, 1]$.
* **Outputs:** A single float value representing the function's response to the specific input coordinate.

## Section 3: Challenge Objectives
The core objective is to maximize response values across eight distinct functions. 
* **Efficiency:** Navigating a limited query budget across all functions simultaneously.
* **Heterogeneity:** Managing diverse function properties, including smooth surfaces, noisy landscapes, and irrelevant dimensions.
* **Dimensionality:** Overcoming the "curse of dimensionality" in high-dimensional spaces (up to 8D) where narrow global peaks are easily missed.

## Section 4: Technical Approach & Strategy Evolution
My strategy has evolved from broad discovery to a **Hierarchical Feature Learning** framework, transitioning from general exploration to surgical refinement.

### Machine Learning Methods
I utilize **Gaussian Process Regression (GPR)** as a surrogate model to map the unknown functions.
* **Kernel Choice:** I employ the **Matern 5/2 kernel**, providing a balance of flexibility and smoothness compared to a standard RBF kernel.
* **Length Scale Optimization:** Using **Automatic Relevance Determination (ARD)**, the model learns the individual influence of each dimension. 
* **Increased Rigor:** To improve model stability, the GPR optimizer restarts have been increased to **40**, ensuring the model avoids local minima when fitting high-dimensional data.

### Hierarchical Search Strategy
In later iterations, a three-tier approach was implemented to organize the optimization logic:
1.  **Local Level (Surgical Strikes):** Pinpointing sharp gradients to reach summits. [cite_start]This led to a peak result of **4440.515** for Function 5 by targeting boundary coordinates[cite: 1, 2].
2.  **Regional Level (Basins of Attraction):** Identifying high-performing neighborhoods. [cite_start]This was critical for navigating the 8D space of Function 8 to maintain competitive results[cite: 2].
3.  **Global Level (Strategic Resets):** Mapping the landscape to identify "Dead Zones." [cite_start]For functions with near-zero signals (e.g., Function 1 at $4.49 \times 10^{-143}$), a "Global Reset" strategy is used to jump to entirely new regions[cite: 2].

### Exploration vs. Exploitation
I utilize an **Adaptive Jitter Map** within the **Expected Improvement (EI)** acquisition function to manage the query budget:
* [cite_start]**Extreme Exploitation ($\xi = 0.0001$):** Used for "locked-in" functions (e.g., Functions 5 and 8) to refine the search around identified peaks[cite: 1, 2].
* [cite_start]**Extreme Exploration ($\xi = 0.2$):** Used to force the model out of negative valleys or zero-signal plateaus[cite: 2].
* **Search Rigor:** To ensure the acquisition optimizer finds the true peak of the surrogate map, I perform **60 restarts** per query.

## Section 5: Repository Architecture
This project follows a modular, research-oriented design:
* **`src/`:** Contains core `BayesianOptimizer` and `Acquisition` modules.
* **`configs/`:** Externalizes hyperparameters (jitter, alpha, restarts) for reproducibility.
* **`data/`:** Maintains cumulative history for each function to ensure accurate model updates with every new query.

***

### Summary of Major Results (Submission 4)
| Function | Dimension | Best Result | Current Strategy |
| :--- | :--- | :--- | :--- |
| 1 | 2D | $4.49 \times 10^{-143}$ | [cite_start]High Jitter Exploration [cite: 2] |
| 5 | 4D | **4440.515** | [cite_start]Boundary Exploitation [cite: 1, 2] |
| 8 | 8D | **9.823** | [cite_start]60-Restart Rigorous Search [cite: 1, 2] |