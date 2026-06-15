# Sequential Model-Based Black-Box Optimization Capstone

## Section 1: Project Overview
This repository contains the complete, final production implementation of a Heterogeneous Bayesian Optimization (HBO) framework designed for the sequential global optimization of unknown, expensive "black-box" functions. 

* **The Core Objective:** To maximize eight hidden engineering functions (F1–F8) spanning varying dimensionalities (2D to 8D) and diverse typographic profiles over a strict 12-round sequential query lifecycle.
* **Real-World Relevance:** This challenge directly mirrors high-stakes industrial engineering and machine learning deployment scenarios—such as hyperparameter tuning for deep neural networks, simulation searches (e.g., Computational Fluid Dynamics), and robotic control policy optimization—where physical or computational trials are heavily resource-constrained.
* **Key Focus:** Shifting engineering priorities from "Big Data" brute-forcing to **"Small Data" operational efficiency**, maximizing information extraction per unit budget under intense spatial uncertainty.

## Section 2: Inputs and Outputs
The optimization pipeline interacts with the hidden evaluation environments via a structured array query protocol:
* **Inputs ($X$):** A continuous $n$-dimensional coordinate vector (ranging from 2D to 8D), where every dimension is mathematically normalized to a unit hypercube bounded strictly within $[0.0, 1.0]$.
* **Outputs ($Y$):** A single floating-point scalar response representing the system's performance metric at that specific coordinate.

## Section 3: Technical Architecture & Strategy Evolution
The engineering pipeline evolved across the 12-round lifecycle from uniform global mapping into an advanced, highly specialized bimodal optimization architecture designed to systematically overcome the curse of dimensionality.

### 1. Statistical Surrogate Modeling
The continuous coordinate landscapes are modeled using **Gaussian Process Regression (GPR)**.
* **Covariance Kernel:** A composite **Constant x Matérn 2.5 kernel** was selected globally to balance surface smoothness with localized flexibility, outperforming standard Radial Basis Function (RBF) setups on complex, narrow ridges.
* **Regularization & Noise Handling:** For deterministic functions, a strict Gaussian noise lower bound ($\alpha = 10^{-6}$) ensures numerical stability. For stochastic landscapes (Function 2), an explicit regularization threshold ($\alpha = 10^{-2}$) is dynamically triggered to prevent the model from overfitting to random environmental noise variance.

### 2. High-Rigor Acquisition Optimization
Sequential query selection is driven by the **Expected Improvement (EI)** acquisition function. To prevent the acquisition optimizer from settling in sub-optimal local traps or flat unmapped regions, the internal search rigor was scaled from an initial baseline of 50 local restarts to a high-rigor maximum of **300 restarts** optimized via the L-BFGS-B algorithm.

### 3. Dynamic Bimodal Jitter ($\xi$) Scheduling
The budget was divided into a specialized investment portfolio using a task-dependent jitter map:
* **Precision Exploitation ($\xi \leq 0.0001$):** Applied to functions with proven, high-performing peaks to force micro-step gradient climbing along sharp summits.
* **Maximum Exploration ($\xi = 0.50$):** Implemented on stagnant or near-zero signal functions to systematically override local surrogate assumptions and force large spatial jumps into unmapped coordinates.

---

## Section 4: Final Milestone Performance Summary

The final phase of the project demonstrated the structural stability and rapid recovery capabilities of the bimodal framework:

| Function | Dimension | Operational Landscape | Max Score | Final Phase Strategy Status |
| :--- | :--- | :--- | :--- | :--- |
| **F1** | 2D | Stagnant Signal | $5.14 \times 10^{-89}$ | Maximum Exploration ($\xi = 0.50$) |
| **F2** | 2D | Stochastic Noisy | **0.2968** | Regularized Exploration ($\alpha = 1e-2, \xi = 0.10$) |
| **F3** | 3D | Non-Responsive | -0.1289 | Maximum Exploration ($\xi = 0.50$) |
| **F4** | 4D | Negative Valley | -0.8791 | Maximum Exploration ($\xi = 0.50$) |
| **F5** | 4D | Volatile / Sharp Peak | **1593.32** | Emergency High-Jitter Reset & Precision Harvest ($\xi = 10^{-4}$) |
| **F6** | 5D | Stagnant | -2.3550 | Maximum Exploration ($\xi = 0.50$) |
| **F7** | 6D | Smooth Ridge | **2.6721** | Pure Exploitation Summit Climb ($\xi = 0.00$) |
| **F8** | 8D | High-Dimensional Peak | **9.7733** | Pure Exploitation 300-Restart Depth Search ($\xi = 0.00$) |

### Major Tactical Highlights: The Function 5 Recovery Case Study
During Submission 10, a premature shift to local precision ($\xi = 0.0001$) caused the acquisition function to stall in a sub-optimal local trap, crashing the output to **16.84**. Recognizing this behavior as a local optimization stall, an **Emergency High-Jitter Reset ($\xi = 0.50$)** was hardcoded into Submission 11 to force the model away from the trap. This successfully recovered the signal back to **1565.23**, setting up the final round for a precision harvest.

---

## Section 5: Repository Architecture
```text

├── function_queries/        # History of query points
├── submission_scripts/      # Historic iteration scripts 
├── notebooks/               # corresponding notebooks
├── README.md                # Comprehensive project documentation
├── ModelCard.md             # Technical documentation and operational constraints
└── DataSet.md               # Data collection lifecycle document