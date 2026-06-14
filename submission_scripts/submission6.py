import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, is_noisy=False):
        # Increased alpha for noisy Function 2 (1e-2), 1e-6 for others
        self.alpha = 1e-2 if is_noisy else 1e-6
        self.model = None

    def load_and_fit(self, func_dir):
        # Loads data including the recently added Submission 5 results
        X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
        Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
        self.X, self.Y, self.dim = X, Y, X.shape[1]
        
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            nu=2.5
        )
        
        # High rigor (50 restarts) to ensure model accuracy in complex spaces
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=self.alpha, 
            normalize_y=True, 
            n_restarts_optimizer=50 
        )
        self.model.fit(self.X, self.Y)

    def expected_improvement(self, x, xi):
        x = x.reshape(-1, self.dim)
        mu, sigma = self.model.predict(x, return_std=True)
        mu_sample_opt = np.max(self.Y)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 0.0] = 0.0
        return ei

    def propose_next_point(self, xi):
        best_x = None
        max_ei = -1
        # 80 restarts to ensure we don't miss the 10.0 peak in Function 8
        for _ in range(80): 
            x0 = np.random.uniform(0.0, 1.0, self.dim)
            res = minimize(lambda x: -self.expected_improvement(x, xi), 
                           x0, 
                           bounds=[(0.0, 1.0)] * self.dim,
                           method='L-BFGS-B')
            if -res.fun > max_ei:
                max_ei = -res.fun
                best_x = res.x
        return best_x

# --- MAIN EXECUTION ---
output_file = "submission_6_results.txt"

# xi_map Logic: "Summit Pursuit" Strategy
# 0.00001: Ultra-Exploitation for Function 8 (Chasing 10.0 milestone)
# 0.0001: Extreme Exploitation for Function 5 (Restoring 4440 peak)
# 0.3: Aggressive Exploration to escape negative/zero regions
xi_map = {
    1: 0.3000, # Still dead zone, need radical jump
    2: 0.0500, # Balanced search for noisy landscape
    3: 0.3000, # Negative, force new region
    4: 0.3000, # Negative, force new region
    5: 0.0001, # EXPLOIT: Returning to the 4440 peak region
    6: 0.3000, # Negative, force new region
    7: 0.0001, # EXPLOIT: Refining the 2.03 peak
    8: 0.00001 # ULTRA-EXPLOIT: Chasing the 10.0 milestone
}

print(f"Generating Submission 6: The Summit Pursuit...")
print("-" * 40)

with open(output_file, "w") as f:
    for i in range(1, 9):
        func_dir = f"function_{i}"
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(func_dir)
        
        current_xi = xi_map[i]
        next_point = optimizer.propose_next_point(xi=current_xi)
        
        # Formatting to match Submission 5 style: Function i: coordinate-coordinate
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        
        result_line = f"Function {i}: {formatted_point}"
        f.write(result_line + "\n")
        print(f"{result_line} (xi={current_xi})")

print("-" * 40)
print(f"Submission 6 generated in {output_file}. Ready for the leaderboard!")