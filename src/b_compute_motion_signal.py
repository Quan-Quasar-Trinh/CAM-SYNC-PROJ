import numpy as np
import matplotlib.pyplot as plt

def compute_motion_signal(keypoints_npy, output_plot="motion_signal.png"):
    keypoints = np.load(keypoints_npy)
    
    velocity = np.diff(keypoints, axis=0)
    
    speed_per_joint = np.linalg.norm(velocity, axis=2)
    
    motion_signal = np.mean(speed_per_joint, axis=1)
    print(motion_signal)
    
    plt.figure(figsize=(12, 6))
    plt.plot(motion_signal)
    plt.title("Motion Signal (Average Joint Velocity)")
    plt.xlabel("Frame")
    plt.ylabel("Average Speed (normalized)")
    plt.grid(True)
    plt.savefig(output_plot)
    plt.show()
    
    print(f"Motion signal shape: {motion_signal.shape}")
    print(f"Đã lưu biểu đồ: {output_plot}")
    
    return motion_signal