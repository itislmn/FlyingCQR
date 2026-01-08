import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def generate_data(n=300):
    X = np.random.uniform(0,np.pi/2, size=n)
    noise = np.random.normal(0, 0.1 * 1.5 * np.abs(X))
    Y = np.sin(X) + noise

    return X.reshape(-1, 1), Y


X, Y = generate_data()

plt.figure()
plt.grid(alpha=0.3)
plt.scatter(X, Y, color="black", s=10, alpha=0.6, label="observations")
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title(r"Heteroscedastic Regression: $y = \sin (x) + \epsilon (x)$", fontsize=13)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("Comparison_Plots/dataset.pdf", format='pdf', dpi=600)
plt.show()
