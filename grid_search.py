import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Generate synthetic "happy student" dataset
np.random.seed(42)
x = np.random.uniform(0, 10, 100)  # Number of courses
true_intercept = 23
true_slope = 3.7
noise = np.random.normal(0, 2, size=x.shape)
y = true_intercept + true_slope * x + noise  # Life satisfaction

# Step 2: Grid search function
def grid_search(x, y, intercept_range, slope_range, resolution):
    intercepts = np.linspace(intercept_range[0], intercept_range[1], resolution)
    slopes = np.linspace(slope_range[0], slope_range[1], resolution)
    sse_grid = np.zeros((resolution, resolution))
    
    for i, intercept in enumerate(intercepts):
        for j, slope in enumerate(slopes):
            y_pred = intercept + slope * x
            sse = np.sum((y - y_pred) ** 2)
            sse_grid[i, j] = sse
    
    min_idx = np.unravel_index(np.argmin(sse_grid), sse_grid.shape)
    best_intercept = intercepts[min_idx[0]]
    best_slope = slopes[min_idx[1]]
    return best_intercept, best_slope, sse_grid, intercepts, slopes

# Step 3: Analytic solution
X = np.vstack([np.ones_like(x), x]).T
beta = np.linalg.inv(X.T @ X) @ X.T @ y
analytic_intercept, analytic_slope = beta

# Step 4: Perform grid search with resolution 50
start_time = time.time()
resolution = 50
best_intercept, best_slope, sse_grid, intercepts, slopes = grid_search(
    x, y, [20, 25], [3.5, 4.0], resolution)
end_time = time.time()

# Print results
print("Analytic result:")
print(f" Intercept: {analytic_intercept:.2f}, slope: {analytic_slope:.2f}")
print("Empirical result:")
print(f" Intercept: {best_intercept:.2f}, slope: {best_slope:.2f}")
print(f"Computation time for resolution {resolution}: {end_time - start_time:.2f} seconds")


# Step 6: Test different resolutions
for resolution in [20, 50, 100, 500]:
    start_time = time.time()
    best_intercept, best_slope, sse_grid, intercepts, slopes = grid_search(
        x, y, [20, 25], [3.5, 4.0], resolution)
    end_time = time.time()
    print(f"Resolution {resolution}: Computation time = {end_time - start_time:.2f} seconds")
    
    # Plot results for current resolution
    plt.figure(figsize=(10, 6))
    plt.imshow(sse_grid, extent=[3.5, 4.0, 20, 25], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Sum of Squared Errors')
    plt.scatter(analytic_slope, analytic_intercept, color='red', label='Analytic Solution', zorder=3)
    plt.scatter(best_slope, best_intercept, color='blue', label='Grid Search Solution', zorder=3)
    plt.xlabel('Slope')
    plt.ylabel('Intercept')
    plt.title(f'Grid Search Results (Resolution {resolution})')
    plt.legend()
    plt.show()