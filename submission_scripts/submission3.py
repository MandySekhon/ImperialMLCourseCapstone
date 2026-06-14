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
        # Load the updated dataset (now contains Initial + Sub 1 + Sub 2)
        X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
        Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
        self.X, self.Y, self.dim = X, Y, X.shape[1]
        
        # Matern 5/2 remains the most robust for these varied landscapes
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            length_scale_bounds=(0.01, 1.0), 
            nu=2.5
        )
        
        # Increased n_restarts_optimizer to 30 for better GP fitting
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=self.alpha, 
            normalize_y=True, 
            n_restarts_optimizer=30
        )
        self.model.fit(self.X, self.Y)

    def expected_improvement(self, x, xi=0.005):
        """
        Refinement: Lowering xi to 0.005 to start 'locking in' on 
        the promising areas found in Round 2.
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

    def propose_next_point(self):
        best_x = None
        max_ei = -1
        # Increased to 40 Restarts to handle the more complex EI surface
        for _ in range(40): 
            x0 = np.random.uniform(0.0, 0.999999, self.dim)
            res = minimize(lambda x: -self.expected_improvement(x), 
                           x0, 
                           bounds=[(0.0, 0.999999)] * self.dim,
                           method='L-BFGS-B')
            if -res.fun > max_ei:
                max_ei = -res.fun
                best_x = res.x
        return best_x

# --- MAIN EXECUTION ---
output_file = "submission_3_results.txt"

print(f"Calculating Round 3 Queries (Refinement Phase)...")
print("-" * 40)

with open(output_file, "w") as f:
    for i in range(1, 9):
        func_dir = f"function_{i}"
        
        # Continue treating Function 2 as the noisy outlier
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(func_dir)
        
        next_point = optimizer.propose_next_point()
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        
        result_line = f"Function {i}: {formatted_point}"
        f.write(result_line + "\n")
        print(result_line)

print("-" * 40)
print(f"Submission 3 generated and saved to {output_file}.")