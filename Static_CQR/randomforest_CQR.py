import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

# reproducibility
RND = 42
np.random.seed(RND)

# number of points
n = 3000

# features: single feature for clear plotting
X = np.random.uniform(-3, 3, size=(n, 1))  # shape (n, 1)

# true function (we'll try to learn this)
def true_function(x):
    return 2.0 * x[:, 0] + np.sin(3.0 * x[:, 0])

# homoskedastic noise (standard normal scaled)
sigma = 0.8
y = true_function(X) + np.random.normal(0, sigma, size=n)

# split: 60% train, 20% calibration (for conformalization), 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=RND)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.25, random_state=RND)
# sizes check
print("train:", X_train.shape, "calib:", X_calib.shape, "test:", X_test.shape)

# Random Forest hyperparameters
n_estimators = 200      # number of trees
max_depth = 8           # depth limit to avoid overfitting and to be fast
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=RND, n_jobs=-1)

# Fit RF on training data (this learns the conditional center/structure)
rf.fit(X_train, y_train)

# helper: get predictions from every tree. returns array shape (n_trees, n_points)
def per_tree_predictions(rf_model, Xq):
    return np.vstack([est.predict(Xq) for est in rf_model.estimators_])

# compute per-tree preds for calibration and test sets
preds_calib_trees = per_tree_predictions(rf, X_calib)  # shape (n_trees, n_calib)
preds_test_trees = per_tree_predictions(rf, X_test)    # shape (n_trees, n_test)

# function to compute quantile from per-tree predictions along axis 0 (trees)
def forest_quantile(preds_by_tree, tau):
    # preds_by_tree shape: (n_trees, n_points)
    # returns vector length n_points with the tau-quantile
    return np.quantile(preds_by_tree, tau, axis=0)

# choose quantile levels we want (e.g. 2.5%, 50%, 97.5%)
taus = [0.025, 0.5, 0.975]

# calibration set quantile estimates
q_calib_lower = forest_quantile(preds_calib_trees, 0.025)
q_calib_median = forest_quantile(preds_calib_trees, 0.5)
q_calib_upper = forest_quantile(preds_calib_trees, 0.975)

# test set quantile estimates
q_test_lower = forest_quantile(preds_test_trees, 0.025)
q_test_median = forest_quantile(preds_test_trees, 0.5)
q_test_upper = forest_quantile(preds_test_trees, 0.975)

# Nonconformity for CQR (from Romano et al.): max(lower - y, y - upper)
nonconformity_calib = np.maximum(q_calib_lower - y_calib, y_calib - q_calib_upper)

# target miscoverage
alpha = 0.05  # e.g., 5% miscoverage -> 95% nominal coverage

# rank-based finite-sample quantile:
n_cal = len(nonconformity_calib)
k = math.ceil((n_cal + 1) * (1 - alpha))
sorted_scores = np.sort(nonconformity_calib)
if k <= n_cal:
    q_conformal = sorted_scores[k - 1]  # index k-1 because of 0-indexing
else:
    q_conformal = sorted_scores[-1]     # fallback: max score
print("q_conformal (additive correction) =", q_conformal)

# Apply additive adjustment to model's conditional interval
interval_lower = q_test_lower - q_conformal
interval_upper = q_test_upper + q_conformal

# interval widths and coverage
interval_widths = interval_upper - interval_lower
coverage = np.mean((y_test >= interval_lower) & (y_test <= interval_upper))
avg_width = np.mean(interval_widths)
mse_median = mean_squared_error(y_test, q_test_median)

print("Coverage (empirical):", coverage)
print("Average width:", avg_width)
print("MSE of median:", mse_median)

# pinball loss implementation (returns positive average loss)
def pinball_loss(y_true, y_pred, tau):
    diff = y_true - y_pred
    # if diff >= 0 -> tau * diff, else -> (tau-1)*diff
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return np.mean(loss)

for tau in taus:
    ypred_tau = forest_quantile(preds_test_trees, tau)
    pl = pinball_loss(y_test, ypred_tau, tau)
    print(f"Pinball loss tau={tau}: {pl:.4f}")


# Sort test points by x
sort_idx = np.argsort(X_test[:, 0])
X_sorted = X_test[sort_idx, 0]
y_sorted = y_test[sort_idx]
median_sorted = q_test_median[sort_idx]
lower_sorted = interval_lower[sort_idx]
upper_sorted = interval_upper[sort_idx]

plt.figure(figsize=(9,5))

# Fill between lower and upper predictions → shaded confidence band
plt.fill_between(
    X_sorted,
    lower_sorted,
    upper_sorted,
    color="green",
    alpha=0.35,
    label="CQR 95% interval"
)
taus = np.random.rand(len(X_test))  # random quantile between 0 and 1
y_pred_sampled = q_test_lower + (q_test_upper - q_test_lower) * taus  # linear interpolation

plt.scatter(X_test[:,0], y_pred_sampled, color="#FDDC5C", s=10, alpha=0.6, label="sampled prediction")


# Median prediction line
plt.plot(X_sorted, median_sorted, color="red", linewidth=2, label="predicted median")

# True data scatter
plt.scatter(X_sorted, y_sorted, color="blue", s=10, alpha=0.6, label="observation")

plt.xlabel("x", fontsize=12)
plt.ylabel("y / prediction", fontsize=12)
plt.title("Conformal Quantile Regression (Random Forest)", fontsize=13)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("static_CQR_Plots/randomforest_CQR.pdf", dpi=600)
plt.show()



