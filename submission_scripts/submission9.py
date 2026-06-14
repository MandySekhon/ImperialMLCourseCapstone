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
        
        # Matern 2.5 remains the best choice for these physical response surfaces
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            nu=2.5
        )
        
        # Maintained 50 restarts for kernel hyperparameter tuning
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
        # Increased to 200 restarts for maximum search depth in the EI surface
        for _ in range(200): 
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
output_file = "submission_9_results.txt"

# xi_map Logic: "Convergence & Recovery"
# F1, F3, F4, F6: Maintain max exploration (0.5) to break out of stagnant regions.
# F5: Increased xi to 0.001 (Recovery). Sub 8 was too narrow; widening search to find 1632+ again.
# F7: Lowered xi to 0.0001 for higher precision refinement of the 2.35 peak.
# F8: Pure exploitation (0.0) with 200 restarts for the final push to 10.0.
xi_map = {
    1: 0.5000, 
    2: 0.1000, 
    3: 0.5000, 
    4: 0.5000, 
    5: 0.0010, 
    6: 0.5000, 
    7: 0.0001, 
    8: 0.0000  
}

print("Generating Submission 9: Convergence & Recovery...")
with open(output_file, "w") as f:
    for i in range(1, 9):
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(f"function_{i}")
        
        next_point = optimizer.propose_next_point(xi=xi_map[i])
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        f.write(f"Function {i}: {formatted_point}\n")
        print(f"Function {i}: {formatted_point} (xi={xi_map[i]})")