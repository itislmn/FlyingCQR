from sklearn.linear_model import QuantileRegressor
from comparison_dataset import X,Y
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.05

qr_lower = QuantileRegressor(quantile=alpha/2, alpha=0)
qr_upper = QuantileRegressor(quantile=1-(alpha/2), alpha=0)

qr_lower.fit(X,Y)
qr_upper.fit(X,Y)
q_lower = qr_lower.predict(X)
q_upper = qr_upper.predict(X)

x = X.flatten()
order = np.argsort(x)
q_x_sorted = x[order]
q_lower_sorted = q_lower[order]
q_upper_sorted = q_upper[order]

plt.figure()
plt.grid(alpha=0.3)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.scatter(X, Y, color="black", s=10, alpha=0.6, label="observations")
plt.plot(X, q_lower, color="purple", label=r"lower quantile regression $\mathcal{Q}^{0.025}$")
plt.plot(X, q_upper, color="purple", label=r"upper quantile regression $\mathcal{Q}^{0.975}$")

plt.fill_between(
    q_x_sorted,
    q_lower_sorted,
    q_upper_sorted,
    color="green",
    alpha=0.3,
    label="Region Prediction (95%)"
)
plt.title("Quantile Regression (QR)")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("Comparison_Plots/quantile_regression.pdf", format='pdf', dpi=600)
plt.show()

