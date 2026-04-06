import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangulate_pose(cam1_kp, cam2_kp, P1, P2):
    """
    cam1_kp, cam2_kp: (33, 2) hoặc (T, 33, 2)
    P1, P2: projection matrix 3x4
    """
    points_3d = []
    
    for i in range(len(cam1_kp)):
        pt1 = cam1_kp[i].reshape(2, 1)
        pt2 = cam2_kp[i].reshape(2, 1)
        
        # Triangulation
        point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
        point_3d = (point_4d[:3] / point_4d[3]).reshape(3,)  # chia homogeneous
        
        points_3d.append(point_3d)
    
    return np.array(points_3d)

def visualize_3d_pose(points_3d, title="3D Pose"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    
    # Kết nối các joint (cơ bản)
    # Bạn có thể dùng mp_pose.POSE_CONNECTIONS để vẽ đầy đủ
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

# Ví dụ Projection matrices (giả định - thay bằng calibration thực tế)
# P1 = K1 @ [I | 0]
# P2 = K2 @ [R | t]
P1 = np.array([[1000, 0, 500, 0],
               [0, 1000, 500, 0],
               [0, 0, 1, 0]], dtype=np.float32)

P2 = np.array([[1000, 0, 500, 100],
               [0, 1000, 500, 0],
               [0, 0, 1, 0]], dtype=np.float32)

# Load keypoints
kp1 = np.load("cam1_keypoints.npy")  # (T, 33, 2)
kp2 = np.load("cam2_keypoints.npy")

# Lấy frame đầu tiên ví dụ
points_3d = triangulate_pose(kp1[0], kp2[0], P1, P2)
visualize_3d_pose(points_3d)