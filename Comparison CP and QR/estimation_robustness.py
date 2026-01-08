import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_clean = 35
clean_data = np.random.normal(loc=-0.2, scale=1, size=n_clean)

#outliers
outliers = np.array([5.5, 6.0, 6.5, 8.5])  # Far right outliers
all_data = np.concatenate([clean_data, outliers])

true_mean = -0.2
sample_mean = np.mean(all_data)
sample_median = np.median(all_data)
ql = np.quantile(all_data, 0.1)
qu = np.quantile(all_data, 0.9)

# Plot
plt.figure(figsize=(10, 6))

plt.scatter(all_data, np.zeros_like(all_data), color='#4c4c4c', marker='o', s=60, label='observations', alpha=0.7)
plt.scatter(outliers, np.zeros_like(outliers), color='red', marker='x', s=100, label='outliers')
plt.axvline(sample_mean, color='#6666ff', linestyle='-', linewidth=2, label='Sample Mean')
plt.axvline(sample_median, color='#efa443', linestyle='-', linewidth=2, label=r'Sample Median')
plt.axvline(true_mean, color='black', linestyle='--', linewidth=2, label='True Mean')
plt.axvline(ql, color='green', linestyle='-', linewidth=2, label=r'Sampled $\mathcal{Q}^{0.1}$ Quantile')
plt.axvline(qu, color='purple', linestyle='-', linewidth=2, label=r'Sampled $\mathcal{Q}^{0.9}$ Quantile')


plt.title('Estimation Robustness', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Density / Count', fontsize=14)
plt.grid(True, alpha=0.6)
plt.legend(loc='upper center', fontsize=12)
plt.ylim(-0.12, 0.12)
plt.xlim(-2.5, 9.0)
plt.tight_layout()
plt.savefig("Comparison_Plots/estimation_robustness.pdf", format='pdf', dpi=600)
plt.show()