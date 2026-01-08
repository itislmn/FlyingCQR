from comparison_dataset import Y
from conformal_prediction import cp_lower, cp_upper
from quantile_regression import q_lower, q_upper
from CQR import cqr_lower, cqr_upper
import numpy as np

def coverage(y, lower, upper):
    return np.mean((y >= lower) & (y <= upper))

print("CP coverage:", coverage(Y, cp_lower, cp_upper))
print("QR coverage:", coverage(Y, q_lower, q_upper))
print("CQR coverage:", coverage(Y, cqr_lower, cqr_upper))