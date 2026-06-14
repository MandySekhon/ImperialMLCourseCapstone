import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, is_noisy=False):
        # We use a higher alpha (1e-2) for Function 2 to handle its "noisy likelihood"
        self.alpha = 1e-2 if is_noisy else 1e-6
        self.model = None

    def load_and_fit(self, func_dir):
        # Load your recently updated dataset (now including Submission 1 results)
        X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
        Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
        self.X, self.Y, self.dim = X, Y, X.shape[1]
        
        # Kernel: Matern 5/2 is less "stiff" than RBF, allowing for real-world irregularities
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim), 
            length_scale_bounds=(0.01, 1.0), 
            nu=2.5
        )
        
        # normalize_y=True is critical because your results range from ~10^-50 to 3486.0
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=self.alpha, 
            normalize_y=True, 
            n_restarts_optimizer=25
        )
        self.model.fit(self.X, self.Y)

    def expected_improvement(self, x, xi=0.01):
        """
        Thinking: xi=0.01 forces the model to look for points significantly 
        better than the current max, preventing "greedy" local sampling.
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
        # 30 Restarts ensure we don't get stuck in a local valley of the acquisition surface
        for _ in range(30): 
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
output_file = "submission_2_results.txt"

print(f"Calculating Round 2 Queries...")
print("-" * 40)

with open(output_file, "w") as f:
    for i in range(1, 9):
        func_dir = f"function_{i}"
        
        optimizer = BayesianOptimizer(is_noisy=(i == 2))
        optimizer.load_and_fit(func_dir)
        
        next_point = optimizer.propose_next_point()
        formatted_point = "-".join([f"{val:.6f}" for val in next_point])
        
        result_line = f"Function {i}: {formatted_point}"
        f.write(result_line + "\n")
        print(result_line)

print("-" * 40)
print(f"Submission 2 generated and saved to {output_file}.")