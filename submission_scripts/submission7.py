import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, is_noisy=False):
        # alpha=1e-2 for noisy F2, 1e-6 for others to maintain high fidelity
        self.alpha = 1e-2 if is_noisy else 1e-6
        self.model = None

    def load_and_fit(self, func_dir):
        X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
        Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
        self.X, self.Y, self.dim = X, Y, X.shape[1]
        
        # Matern nu=2.5 is the standard for modeling physical/non-smooth processes
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            nu=2.5
        )
        
        # 50 restarts to ensure the length scales (ARD) are perfectly tuned
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
        # 100 restarts for the acquisition function to find the global EI peak
        for _ in range(100): 
            x0 = np.random.uniform(0.0, 1.0, self.dim)
            res = minimize(lambda x: -self.expected_improvement(x, xi), 
                           x0, 
                           bounds=[(0.0, 1.0)] * self.dim,
                           method='L-BFGS-B')
            if -res.fun > max_ei:
                max_ei = -res.fun
                best_x = res.x
        return best_x

# --- EXECUTION LOGIC ---
output_file = "submission_7_results.txt"

# xi_map Logic: "Surgical Calibration"
# F1, F3, F4, F6: Increased xi to 0.4 to force a jump out of deep negative zones.
# F5: Decreased xi to 0.00005 to hyper-refine the 790/4440 basin.
# F8: xi = 0.0 (Pure Exploitation). We are at 9.914; it's time to hit 10.0.
xi_map = {
    1: 0.4000, 
    2: 0.0500, 
    3: 0.4000, 
    4: 0.4000, 
    5: 0.00005, 
    6: 0.4000, 
    7: 0.0001, 
    8: 0.0000  
}

print("Generating Submission 7: Surgical Calibration...")
with open(output_file, "w") as f:
    for i in range(1, 9):
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(f"function_{i}")
        
        next_point = optimizer.propose_next_point(xi=xi_map[i])
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        f.write(f"Function {i}: {formatted_point}\n")
        print(f"Function {i}: {formatted_point} (xi={xi_map[i]})")