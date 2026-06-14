import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, is_noisy=False):
        # Alpha 1e-2 for noisy Function 2, 1e-6 for others to trust data points
        self.alpha = 1e-2 if is_noisy else 1e-6
        self.model = None

    def load_and_fit(self, func_dir):
        # Load cumulative data (Initial + Sub 1 + Sub 2 + Sub 3 + Sub 4)
        X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
        Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
        self.X, self.Y, self.dim = X, Y, X.shape[1]
        
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            length_scale_bounds=(0.01, 10.0), # Expanded bounds for Sub 5
            nu=2.5
        )
        
        # Increased to 40 restarts for maximum model accuracy
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=self.alpha, 
            normalize_y=True, 
            n_restarts_optimizer=40
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
        # 60 restarts to ensure we don't miss the 8D peak in Function 8
        for _ in range(60): 
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
output_file = "submission_5_results.txt"

# xi_map Logic:
# 0.0001: Extreme Exploitation (Locking in massive wins)
# 0.01: Standard balanced search
# 0.2: Extreme Exploration (Escaping negative valleys)
xi_map = {
    1: 0.2000, # Still near zero, need a huge jump
    2: 0.0100, # Steady climb
    3: 0.2000, # Failed origin probe, searching elsewhere
    4: 0.2000, # Stuck in negative territory
    5: 0.0001, # EXPLOIT: Found huge peak (4440.5)
    6: 0.2000, # Dropped significantly, need reset
    7: 0.0001, # EXPLOIT: Near peak (2.0+)
    8: 0.0001  # EXPLOIT: Near peak (9.8+)
}

print(f"Generating Submission 5: The Final Climb...")
print("-" * 40)

with open(output_file, "w") as f:
    for i in range(1, 9):
        func_dir = f"function_{i}"
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(func_dir)
        
        current_xi = xi_map[i]
        next_point = optimizer.propose_next_point(xi=current_xi)
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        
        result_line = f"Function {i}: {formatted_point}"
        f.write(result_line + "\n")
        print(f"{result_line} (xi={current_xi})")

print("-" * 40)
print(f"Submission 5 generated. Good luck on the leaderboard!")