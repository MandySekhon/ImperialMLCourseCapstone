---

### 3. Final Production `DataSet.md`

```markdown
# Datasheet: Black-Box Optimization Capstone Project Data Set

This datasheet follows the standard structured framework for documenting machine learning data sets, ensuring reproducibility, transparent data lineage, and ethical data governance.

## 1. Motivation
* **Purpose:** This data set was sequentially generated to benchmark and solve a multi-task Black-Box Optimization (BBO) problem, specifically designed for an advanced machine learning engineering and data science capstone.
* **Task Support:** The data directly supports sequential model-based optimization tasks, specifically tracking the performance of Gaussian Process Regression (GPR) and Expected Improvement (EI) algorithms in sample-starved environments.

## 2. Composition and Structure
* **Contents:** The data set contains sequential input-output coordinate pairs for eight distinct optimization tasks (F1–F8).
  * **Inputs ($X$):** Continuous coordinates formatted as NumPy float64 arrays. Dimensionalities vary by task: F1-F2 (2D), F3 (3D), F4-F5 (4D), F6 (5D), F7 (6D), F8 (8D). All input dimensions are strictly bounded within the normalized unit hypercube $[0.0, 1.0]$.
  * **Outputs ($Y$):** Single scalar floating-point responses returned directly by the hidden testing environments.
* **Data Volume:** Contains 12 discrete sequential iterations of sampled data across all 8 functions, building upon the initial seed arrays.
* **Data Omissions & Sampling Gaps:** The data set contains a deliberate and significant exploitation bias within the high-performing domains (F5, F7, F8). Because late-stage resources were redirected to maximize final scalar rewards along proven paths, massive volumes of the high-dimensional hypercube spaces remain completely unsampled.

## 3. Collection and Generation Methodology
* **Generation Framework:** Data points were collected sequentially through a tight closed-loop Bayesian Optimization workflow:

```text
[Surrogate fitting (GPR)] ──> [Acquisition Optimization] ──> [Query Generation] ──> [Environment Feedback]