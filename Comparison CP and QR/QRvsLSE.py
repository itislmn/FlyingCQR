import numpy as np
import matplotlib.pyplot as plt

# Define loss functions
def least_squares_loss(residual):
    return residual ** 2

def quantile_loss(residual, tau):
    return np.where(residual >= 0, tau * residual, (tau - 1) * residual)

# Generate residuals
residuals = np.linspace(-1.0, 1.0, 100)

# Compute losses
ls_loss = least_squares_loss(residuals)
q01_loss = quantile_loss(residuals, 0.1)
q05_loss = quantile_loss(residuals, 0.5)
q09_loss = quantile_loss(residuals, 0.9)

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(residuals, ls_loss, color="#6666ff", marker='*', markevery=10, label='Least Squares',linewidth=3)
plt.plot(residuals, q01_loss, color="#ea4848" , marker='s', markevery=10, label=r'$\mathcal{Q}^{0.1}$ Quantile', linewidth=3)
plt.plot(residuals, q05_loss, color="#efa443", marker='s', markevery=10, label=r'$\mathcal{Q}^{0.5}$ Median', linewidth=3)
plt.plot(residuals, q09_loss, color="#edd02b", marker='s', markevery=10, label=r'$\mathcal{Q}^{0.9}$ Quantile', linewidth=3)

plt.title('Regression: Loss Function', fontsize=16)
plt.xlabel('Residual', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, alpha=0.6)
plt.legend(loc='upper center', fontsize=12)
plt.xlim(-1.0, 1.0)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("Comparison_Plots/regression_loss.pdf", format='pdf', dpi=600)
plt.show()