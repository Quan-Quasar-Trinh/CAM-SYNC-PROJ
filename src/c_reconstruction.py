import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def synchronize_keypoints(kp1, kp2):
    """
    Đồng bộ keypoints giữa 2 camera theo frame index (giả lập)
    """

    print(f"Trước khi sync:")
    print(f"   Cam1: {kp1.shape}")
    print(f"   Cam2: {kp2.shape}")
    
    # === GIẢ LẬP ĐỒNG BỘ THEO FRAME INDEX ===
    min_frames = min(len(kp1), len(kp2))
    
    # Cắt cả hai về cùng số frame (phương pháp đơn giản nhất)
    kp1_sync = kp1[:min_frames]
    kp2_sync = kp2[:min_frames]
    
    
    print(f"\n✅ Đã đồng bộ thành công!")
    print(f"   Số frame sau sync: {min_frames}")
    
    return kp1_sync, kp2_sync


def triangulate_pose(cam1_kp, cam2_kp, P1, P2):
    """
    cam1_kp, cam2_kp: (33, 2) for single frame or (T, 33, 2) for sequence
    P1, P2: projection matrix 3x4
    """
    if cam1_kp.ndim == 2:  
        points_3d = []
        for i in range(len(cam1_kp)):
            pt1 = cam1_kp[i].reshape(2, 1)
            pt2 = cam2_kp[i].reshape(2, 1)
            
            point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
            point_3d = (point_4d[:3] / point_4d[3]).reshape(3,)
            
            points_3d.append(point_3d)
        
        return np.array(points_3d)
    
    elif cam1_kp.ndim == 3: 
        T = cam1_kp.shape[0]
        points_3d = []
        for t in range(T):
            frame_points = []
            for i in range(cam1_kp.shape[1]):
                pt1 = cam1_kp[t, i].reshape(2, 1)
                pt2 = cam2_kp[t, i].reshape(2, 1)
                
                point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
                point_3d = (point_4d[:3] / point_4d[3]).reshape(3,)
                
                frame_points.append(point_3d)
            
            points_3d.append(frame_points)
        
        return np.array(points_3d)
    
    else:
        raise ValueError("Invalid shape for cam1_kp. Expected (33, 2) or (T, 33, 2)")

def visualize_3d_pose(points_3d, title="3D Pose"):
    if points_3d.ndim == 2: 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()
    
    elif points_3d.ndim == 3:  
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            points = points_3d[frame]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} - Frame {frame}')
        
        ani = animation.FuncAnimation(fig, update, frames=points_3d.shape[0], interval=100)
        plt.show()
    
    else:
        raise ValueError("Invalid shape for points_3d. Expected (33, 3) or (T, 33, 3)")


P1 = np.array([[1000, 0, 500, 0],
               [0, 1000, 500, 0],
               [0, 0, 1, 0]], dtype=np.float32)

P2 = np.array([[1000, 0, 500, 100],
               [0, 1000, 500, 0],
               [0, 0, 1, 0]], dtype=np.float32)
if __name__ == "__main__":
    kp1 = np.load("cam1_keypoints.npy")
    kp2 = np.load("cam2_keypoints.npy")

    points_3d = triangulate_pose(kp1[0], kp2[0], P1, P2)
    visualize_3d_pose(points_3d)