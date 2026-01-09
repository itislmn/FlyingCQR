import cv2
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pykalman import KalmanFilter

# -----------------------------
# 1️⃣ Load video and track pigeon
# -----------------------------
video_path = r"C:\Users\hossa\Documents\Work\TU Darmstadt\Semester 4\Projektseminar KTS\Conformal Prediction - Thesis\pigeon.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

frame0 = cv2.resize(frame0, (frame0.shape[1], frame0.shape[0]))  # keep original size or scale down if needed

cv2.namedWindow("Select ONE pigeon", cv2.WINDOW_NORMAL)
bbox = cv2.selectROI("Select ONE pigeon", frame0, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

tracker = cv2.TrackerCSRT_create()
tracker.init(frame0, bbox)

trajectory = []
trajectory.append([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2])  # first frame

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(1, n_frames):
    ret, frame = cap.read()
    if not ret:
        break

    ok, bbox = tracker.update(frame)
    if ok:
        cx = bbox[0] + bbox[2]/2
        cy = bbox[1] + bbox[3]/2
        trajectory.append([cx, cy])
    else:
        trajectory.append([np.nan, np.nan])

trajectory = np.array(trajectory)
n = len(trajectory)
print(f"Tracked {n} frames")

cap.release()

# -----------------------------
# 2️⃣ Kalman Filter for trajectory
# -----------------------------
dt = 1.0
A = np.array([[1,0,dt,0],
              [0,1,0,dt],
              [0,0,1,0],
              [0,0,0,1]])
C = np.array([[1,0,0,0],
              [0,1,0,0]])

kf = KalmanFilter(
    transition_matrices=A,
    observation_matrices=C,
    initial_state_mean=[trajectory[0,0], trajectory[0,1], 0,0],
    observation_covariance=np.eye(2)*5,
    transition_covariance=np.eye(4)
)

state_means, state_covs = kf.filter(trajectory)
innovations = trajectory - np.dot(C, state_means.T).T

# -----------------------------
# 3️⃣ KalmanCQR: Quantile regression + conformal calibration
# -----------------------------
train_end = int(0.8 * n)
calib_end = int(0.9 * n)
alpha = 0.10
t = np.arange(n).reshape(-1,1)
X = sm.add_constant(t)

lower_bounds = np.zeros_like(trajectory)
upper_bounds = np.zeros_like(trajectory)

# Start prediction intervals at 65% of the video
start_pred = int(0.65 * n)

# Scaling and minimum PI for visibility
scale = 10
min_pi = 3

for dim in range(2):
    qr_lo = sm.QuantReg(innovations[:,dim], X).fit(q=alpha/2)
    qr_hi = sm.QuantReg(innovations[:,dim], X).fit(q=1-alpha/2)

    q_lo = qr_lo.predict(X)
    q_hi = qr_hi.predict(X)

    innov_slice = innovations[train_end:calib_end, dim]
    q_lo_slice = q_lo[train_end:calib_end]
    q_hi_slice = q_hi[train_end:calib_end]

    cal_scores = np.maximum(
        q_lo_slice - innov_slice,
        innov_slice - q_hi_slice
    )
    q_hat = np.quantile(cal_scores, 1-alpha)

    # Fill prediction bounds starting at 65% of the video
    lower_bounds[start_pred:, dim] = q_lo[start_pred:] - q_hat
    upper_bounds[start_pred:, dim] = q_hi[start_pred:] + q_hat

# -----------------------------
# 4️⃣ Plot trajectory + prediction intervals
# -----------------------------
plt.figure(figsize=(6,6))
plt.plot(trajectory[:,0], trajectory[:,1], "-o", markersize=2, color="black", label="Tracked Trajectory")

for i in range(start_pred, n):
    rx = max((upper_bounds[i,0]-lower_bounds[i,0])*scale, min_pi)
    ry = max((upper_bounds[i,1]-lower_bounds[i,1])*scale, min_pi)
    ellipse = plt.Circle((trajectory[i,0], trajectory[i,1]), max(rx, ry), color="red", alpha=0.2)
    plt.gca().add_patch(ellipse)

plt.gca().invert_yaxis()
plt.title("Tracked Pigeon Trajectory with KalmanCQR Prediction Region")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("trajectory.pdf", format="pdf", dpi=600)
plt.show()

# -----------------------------
# 5️⃣ Prediction interval width over time
# -----------------------------
pi_width = np.sqrt((upper_bounds[:,0]-lower_bounds[:,0])**2 + (upper_bounds[:,1]-lower_bounds[:,1])**2)
plt.figure(figsize=(8,3))
plt.plot(pi_width, color="red")
plt.title("Prediction Interval Width Over Time")
plt.xlabel("Frame")
plt.ylabel("PI width (pixels)")
plt.grid(alpha=0.3)
plt.savefig("kalmanPI_bird.pdf", format="pdf", dpi=600)
plt.show()

# -----------------------------
# 6️⃣ Video overlay with green dot always, ellipse after 65%
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video = cv2.VideoWriter(
    'pigeon_KalmanCQR_start65_fixed.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

for i in range(n):
    ret, frame = cap.read()
    if not ret:
        break

    x, y = trajectory[i]

    if not np.isnan(x):
        # ✅ Always draw tracked pigeon (green dot)
        cv2.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

        # ✅ Draw prediction interval ellipse ONLY after start_pred
        if i >= start_pred:
            rx = max(int((upper_bounds[i,0]-lower_bounds[i,0])*scale), min_pi)
            ry = max(int((upper_bounds[i,1]-lower_bounds[i,1])*scale), min_pi)

            overlay = frame.copy()
            cv2.ellipse(
                overlay,
                (int(x), int(y)),
                (rx, ry),
                0, 0, 360,
                (0,0,255), 2
            )
            alpha_ellipse = 0.3
            frame = cv2.addWeighted(overlay, alpha_ellipse, frame, 1-alpha_ellipse, 0)

    out_video.write(frame)

cap.release()
out_video.release()
print("✅ KalmanCQR video saved as 'pigeon_KalmanCQR.mp4'")
