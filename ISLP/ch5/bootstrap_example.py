import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data

# Load the Portfolio dataset
Portfolio = load_data('Portfolio')

# Define the function to calculate alpha from the data
def alpha_func(D, idx):
    """
    Calculate the minimum variance portfolio alpha
    Formula: α = (σ_Y² - σ_XY) / (σ_X² + σ_Y² - 2σ_XY)
    """
    # Calculate covariance matrix for columns X and Y
    cov_ = np.cov(D[['X','Y']].loc[idx], rowvar=False)
    return ((cov_[1,1] - cov_[0,1]) / 
            (cov_[0,0] + cov_[1,1] - 2*cov_[0,1]))

# Bootstrap function to estimate standard error
def boot_SE(func, D, n=None, B=1000, seed=0):
    """
    Calculate bootstrap standard error for a statistic
    
    Parameters:
    func - function that calculates the statistic
    D - DataFrame containing the data
    n - sample size (defaults to size of D)
    B - number of bootstrap samples
    seed - random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    values = []  # Store calculated statistic for each bootstrap sample
    
    n = n or D.shape[0]  # Use full dataset size if n not specified
    
    for _ in range(B):
        # Create bootstrap sample by sampling with replacement
        idx = rng.choice(D.index, n, replace=True)
        # Calculate statistic for this bootstrap sample
        value = func(D, idx)
        values.append(value)
    
    # Calculate standard error as std deviation of bootstrap estimates
    return np.std(values, ddof=1), values

# Calculate alpha for the full dataset
alpha_full = alpha_func(Portfolio, range(100))
print(f"Alpha estimate using all data: {alpha_full:.4f}")

# Calculate bootstrap standard error and get all bootstrap values
alpha_SE, bootstrap_values = boot_SE(alpha_func, Portfolio, B=1000, seed=0)
print(f"Bootstrap estimate of standard error: {alpha_SE:.4f}")

# Generate 20 bootstrap samples and save results
rng = np.random.default_rng(0)
bootstrap_alphas = []
for i in range(20):
    idx = rng.choice(100, 100, replace=True)
    alpha_bs = alpha_func(Portfolio, idx)
    bootstrap_alphas.append(alpha_bs)
    print(f"Bootstrap sample {i+1}: alpha = {alpha_bs:.4f}")

# Create histogram of bootstrap results
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_values, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(alpha_full, color='red', linestyle='dashed', linewidth=2, label=f'Full Data Alpha: {alpha_full:.4f}')
plt.axvline(np.mean(bootstrap_values), color='green', linestyle='dashed', linewidth=2, 
            label=f'Bootstrap Mean: {np.mean(bootstrap_values):.4f}')
plt.xlabel('Alpha Value')
plt.ylabel('Frequency')
plt.title('Distribution of Alpha Estimates from 1000 Bootstrap Samples')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print summary statistics
print(f"\nSummary Statistics for 1000 Bootstrap Samples:")
print(f"Mean: {np.mean(bootstrap_values):.4f}")
print(f"Standard Error: {np.std(bootstrap_values, ddof=1):.4f}")
print(f"95% Confidence Interval: ({np.percentile(bootstrap_values, 2.5):.4f}, {np.percentile(bootstrap_values, 97.5):.4f})")