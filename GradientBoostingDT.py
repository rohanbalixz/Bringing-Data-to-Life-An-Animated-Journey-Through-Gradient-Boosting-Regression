import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic regression data
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Gradient Boosting Regressor with 300 estimators
n_estimators = 500
gbr = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1,
                                max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# For smooth plotting, sort the test set by feature values
sorted_idx = np.argsort(X_test.ravel())
X_test_sorted = X_test[sorted_idx]
y_test_sorted = y_test[sorted_idx]

# Get staged predictions on the test set
staged_preds = list(gbr.staged_predict(X_test_sorted))

# Pre-compute performance metrics for each iteration
r2_list = []
mse_list = []
for pred in staged_preds:
    r2_list.append(r2_score(y_test_sorted, pred))
    mse_list.append(mean_squared_error(y_test_sorted, pred))

# Set up the plot: a single graph for predictions and metrics
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test_sorted, y_test_sorted, color='blue', label='Test Data')
pred_line, = ax.plot(X_test_sorted, staged_preds[0], 'r-', lw=2, label='Model Prediction')
metrics_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.set_title('Gradient Boosting Regressor Predictions with Metrics')
ax.legend()

def animate(i):
    # Update the prediction curve
    pred_line.set_ydata(staged_preds[i])
    # Update the metrics text with current iteration's R-squared and MSE values
    current_r2 = r2_list[i]
    metrics_text.set_text(f"Iteration: {i}\nR²: {current_r2:.3f}")
    return pred_line, metrics_text

# Create the animation (update every 50ms)
ani = animation.FuncAnimation(fig, animate, frames=len(staged_preds), interval=50, blit=True)

plt.show()
