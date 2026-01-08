from conformal_prediction import X_train, Y_train, X_calib, Y_calib
from comparison_dataset import X, Y
from sklearn.linear_model import QuantileRegressor
from quantile_regression import q_lower, q_upper
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.05

qr_l = QuantileRegressor(quantile=alpha/2, alpha=0)
qr_u = QuantileRegressor(quantile=1-(alpha/2), alpha=0)

qr_l.fit(X_train, Y_train)
qr_u.fit(X_train, Y_train)

calib_scores = np.maximum(qr_l.predict(X_calib) - Y_calib, Y_calib - qr_u.predict(X_calib))

q_cqr = np.quantile(calib_scores, 1 - alpha)

cqr_lower = qr_l.predict(X) - q_cqr
cqr_upper = qr_u.predict(X) + q_cqr

x = X.flatten()
order = np.argsort(x)
cqr_x_sorted = x[order]
cqr_l_sorted = cqr_lower[order]
cqr_u_sorted = cqr_upper[order]

plt.figure()
plt.grid(alpha=0.3)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.scatter(X_train, Y_train, color="blue", s=10, alpha=0.6, label="observations (test)")
plt.scatter(X_calib, Y_calib, color="red", s=10, alpha=0.6, label="observations (calibration)")
plt.plot(X, q_lower, color="purple", label=r"lower quantile regression $\mathcal{Q}^{0.025}$")
plt.plot(X, q_upper, color="purple", label=r"upper quantile regression $\mathcal{Q}^{0.975}$")
plt.fill_between(
    cqr_x_sorted,
    cqr_l_sorted,
    cqr_u_sorted,
    color="green",
    alpha=0.3,
    label="Region Prediction (95%)"
)
plt.title("Conformalized Quantile Regression (CQR)")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("Comparison_Plots/cqr.pdf", format='pdf', dpi=600)
plt.show()