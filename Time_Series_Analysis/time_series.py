import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import QuantileRegressor
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL

#AirPassengers data (1949-01 to 1960-12) — 144 monthly values
air_passengers_data = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
]

# Create time series with datetime index
dates = pd.date_range(start='1949-01', periods=len(air_passengers_data), freq='M')
y = pd.Series(air_passengers_data, index=dates, name='passengers')

# Plot original data
plt.figure(figsize=(10, 3))
plt.plot(y, color='black', alpha=0.75)
plt.title('AirPassengers: Monthly International Airline Passengers (1949–1960)')
plt.ylabel(r'Passengers ($\ast10^3$)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Fit SARIMA(0,1,1)x(0,1,1)[12]
print("Fitting SARIMA(0,1,1)x(0,1,1)[12]...")
sarima_model = ARIMA(
    y,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12)  # ← Fixed: 12, not "1 2"
)
sarima_fit = sarima_model.fit()

# Get fitted values and residuals
y_pred = sarima_fit.fittedvalues
residuals = y - y_pred

# Plot SARIMA fit and residuals
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(y, label='Observed', color='black', alpha=0.75)
axs[0].plot(y_pred, label='ARIMA Forecast', color='red')
axs[0].set_ylabel('Passengers')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(residuals, color='purple')
axs[1].set_ylabel('Residuals')
axs[1].set_xlabel('Year')
axs[1].grid(True, alpha=0.3)
plt.suptitle('ARIMA Fit and Residuals')
plt.tight_layout()
plt.savefig(r'Time_Series_Plots/AirPassengers_residuals.pdf', format='pdf', dpi=600)
plt.show()


stl = STL(y, period=12, robust=True)
res = stl.fit()
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(res.trend, label='Trend', color='darkblue', alpha=0.75)
axs[0].legend()
axs[1].plot(res.seasonal, label='Seasonality', color='darkred', alpha=0.75)
axs[1].legend()
for ax in axs:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'Time_Series_Plots/AirPassengers_STL.pdf', format='pdf', dpi=600)
plt.show()

# Residual diagnostics
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_acf(residuals.dropna(), ax=ax[0], lags=20, alpha=0.05, color='darkblue')
plot_pacf(residuals.dropna(), ax=ax[1], lags=20, alpha=0.05, color='darkred')

for coll in ax[0].collections:
    coll.set_facecolor('navy')
    coll.set_alpha(0.3)

for coll in ax[1].collections:
    coll.set_facecolor('darkorange')
    coll.set_alpha(0.3)

ax[0].set_title('ACF of Residuals')
ax[1].set_title('PACF of Residuals')
plt.tight_layout()
plt.savefig('Time_Series_Plots/AirPassengers_corr.pdf', format='pdf', dpi=600)
plt.show()

# Ljung-Box test
lb_result = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
print("Ljung-Box test (H₀: residuals uncorrelated):")
print(lb_result)

# Conformalized Quantile Regression on residuals
time_idx = np.arange(len(y)).reshape(-1, 1)
valid = ~residuals.isna()
X = time_idx[valid]
r = residuals[valid].values


n = len(r)
train_end = int(0.8 * n)
calib_end = int(0.9 * n)

X_train, r_train = X[:train_end], r[:train_end]
X_calib, r_calib = X[train_end:calib_end], r[train_end:calib_end]
X_test, r_test = X[calib_end:], r[calib_end:]
y_test = y[valid][calib_end:].values

# Fit quantile regressors
alpha = 0.05
qr_low = QuantileRegressor(quantile=alpha/2, alpha=0, solver='highs').fit(X_train, r_train)
qr_up  = QuantileRegressor(quantile=1 - alpha/2, alpha=0, solver='highs').fit(X_train, r_train)

# Calibration scores
q_low_calib = qr_low.predict(X_calib)
q_up_calib = qr_up.predict(X_calib)
scores = np.maximum(r_calib - q_up_calib, q_low_calib - r_calib)
qhat = np.quantile(scores, 1 - alpha)

# Prediction intervals on test set
y_pred_test = y_pred[valid][calib_end:].values
q_low_test = qr_low.predict(X_test)
q_up_test = qr_up.predict(X_test)
lower = y_pred_test + q_low_test - qhat
upper = y_pred_test + q_up_test + qhat

coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"\nConformal adjustment (q̂): {qhat:.2f}")
print(f"Empirical coverage: {coverage:.2%} (target: {1-alpha:.0%})")

# Final plot with conformal intervals
plt.figure(figsize=(10, 4))
plt.plot(y, label='True', color='black')
plt.plot(y_pred, label='ARIMA Point Forecast', color='red')
plt.fill_between(
    y.index[valid][calib_end:], lower, upper,
    color='green', alpha=0.3,
    label=f'{int((1-alpha)*100)}% Prediction Interval'
)
plt.axvline(y.index[valid][calib_end], color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.title('ARIMA + CQR: Uncertainty Quantification for Time Series')
plt.ylabel(r'Passengers ($\ast10^3$)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Time_Series_Plots/AirPassengers.pdf', format='pdf', dpi=600)
plt.show()