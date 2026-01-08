from sklearn.linear_model import LinearRegression
from comparison_dataset import X, Y
import numpy as np
import matplotlib.pyplot as plt


#split
n= len(X)
idx = np.random.permutation(n)
train_idx, calib_idx, = idx[:n//2], idx[n//2:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_calib, Y_calib = X[calib_idx], Y[calib_idx]

model = LinearRegression()
model.fit(X_train, Y_train)

residuals = np.abs(Y_calib - model.predict(X_calib))

alpha = 0.05
q_hat = np.quantile(residuals, 1 - alpha)


y_pred = model.predict(X)
cp_lower = y_pred - q_hat
cp_upper = y_pred + q_hat

x = X.flatten()
order = np.argsort(x)
cp_x_sorted = x[order]
cp_lower_sorted = cp_lower[order]
cp_upper_sorted = cp_upper[order]

plt.figure()
plt.grid(alpha=0.3)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.scatter(X_train, Y_train, color="blue", s=10, alpha=0.6, label="observations (test)")
plt.scatter(X_calib, Y_calib, color="red", s=10, alpha=0.6, label="observations (calibration)")
plt.fill_between(
    cp_x_sorted,
    cp_lower_sorted,
    cp_upper_sorted,
    color="green",
    alpha=0.35,
    label="Region Prediction (95%)"
)
plt.title("Conformal Prediciton (CP)")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("Comparison_Plots/conformal_prediction.pdf", format='pdf', dpi=600)
plt.show()

