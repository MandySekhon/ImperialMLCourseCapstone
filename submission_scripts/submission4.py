import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, is_noisy=False):
        # Function 2 remains noisy (alpha=1e-2), others remain clean (1e-6)
        self.alpha = 1e-2 if is_noisy else 1e-6
        self.model = None

    def load_and_fit(self, func_dir):
        # Load cumulative dataset (Initial + Sub 1 + Sub 2 + Sub 3)
        X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
        Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
        self.X, self.Y, self.dim = X, Y, X.shape[1]
        
        # Matern 5/2 remains the most robust for non-linear landscapes
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            length_scale_bounds=(0.01, 1.0), 
            nu=2.5
        )
        
        # Increased n_restarts_optimizer to 35 for higher fitting precision
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=self.alpha, 
            normalize_y=True, 
            n_restarts_optimizer=35
        )
        self.model.fit(self.X, self.Y)

    def expected_improvement(self, x, xi):
        """
        Calculates Expected Improvement with dynamic jitter (xi).
        """
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
        # Increased to 50 Restarts to ensure we find the global maximum of the EI surface
        for _ in range(50): 
            x0 = np.random.uniform(0.0, 0.999999, self.dim)
            res = minimize(lambda x: -self.expected_improvement(x, xi), 
                           x0, 
                           bounds=[(0.0, 0.999999)] * self.dim,
                           method='L-BFGS-B')
            if -res.fun > max_ei:
                max_ei = -res.fun
                best_x = res.x
        return best_x

# --- MAIN EXECUTION ---
output_file = "submission_4_results.txt"

# Dynamic Jitter Strategy:
# Low xi (0.001) for SUCCESS: Exploiting peaks found in Sub 3
# High xi (0.1) for STUCK: Forcing exploration for negative/zero results
# Medium xi (0.01) for STEADY: Balanced search
xi_map = {
    1: 0.100,  # Stuck at 10^-232, need massive exploration 
    2: 0.010,  # Steady progress 
    3: 0.100,  # Negative result, need to move 
    4: 0.100,  # Negative result, need to move 
    5: 0.001,  # EXPLOIT: Found huge peak (4440.48) 
    6: 0.100,  # Negative result, need to move 
    7: 0.001,  # EXPLOIT: Consistent growth 
    8: 0.001   # EXPLOIT: Found high value (9.88) 
}

print(f"Calculating Round 4 Queries (Climbing & Exploration Phase)...")
print("-" * 40)

with open(output_file, "w") as f:
    for i in range(1, 9):
        func_dir = f"function_{i}"
        
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(func_dir)
        
        # Use the function-specific jitter value
        current_xi = xi_map[i]
        next_point = optimizer.propose_next_point(xi=current_xi)
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        
        result_line = f"Function {i}: {formatted_point}"
        f.write(result_line + "\n")
        print(f"{result_line} (xi={current_xi})")

print("-" * 40)
print(f"Submission 4 generated and saved to {output_file}.")