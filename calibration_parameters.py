import numpy as np

# Camera parameters from your YAML file
K = np.array([[904.2736206054688, 0.0, 631.90087890625],
              [0.0, 904.5254516601562, 375.46514892578125],
              [0.0, 0.0, 1.0]])

D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Distortion coefficients
R = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])  # Rotation matrix
P = np.array([[904.2736206054688, 0.0, 631.90087890625, 0.0],
              [0.0, 904.5254516601562, 375.46514892578125, 0.0],
              [0.0, 0.0, 1.0, 0.0]])

# Save to .npz file
np.savez("camera_calibration.npz", camera_matrix=K, dist_coeffs=D, rotation_matrix=R, projection_matrix=P)

