### 2. Final Production `ModelCard.md`

```markdown
# Model Card: Heterogeneous Bayesian Optimization (HBO)

This model card provides technical documentation for the Gaussian Process-backed sequential optimization architecture deployed during the engineering capstone project, supporting transparent and reproducible AI governance.

## 1. Model Details
* **Model Name:** Heterogeneous Bayesian Optimizer (HBO)
* **Model Architecture:** Gaussian Process Regression (GPR) Surrogate combined with an Expected Improvement (EI) Acquisition Function
* **Production Version:** v12.0 (Final Deployment Status)
* **Underlying Libraries:** `scikit-learn` (v1.4+), `scipy` (v1.12+)

## 2. Intended Use
* **Primary Task:** Global optimization of unmapped, expensive black-box mathematical systems under strict evaluation limits.
* **Target Environment:** Multi-task settings where target functions exhibit highly heterogeneous behaviors (e.g., highly smooth continuous profiles mixed with high-dimensional spaces or stochastic noise).
* **Scenarios to Avoid:** Objective functions containing discontinuous step changes or discrete jumps, as the GPR spatial covariance kernel assume continuous differentiability.

## 3. Technical Design and Evolution
The framework transitioned through three distinct operational phases over its 12-week lifecycle:

```text
[Rounds 1-8: Uniform Base]      [Rounds 9-10: Scaled Rigor]       [Rounds 11-12: Bimodal Harvest]
  - xi uniformly at 0.01          - Restart Depth to 200/300        - Stagnant: Max Jitter (0.50)
  - 50 Optimizer Restarts         - Specialized Noisy F2 Alpha      - Peaks: Pure Exploitation (0.00)
  - Coarse Surface Mapping        - Vulnerable to Local Traps       - Emergency Recovery Systems