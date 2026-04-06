import cv2
import numpy as np
import os
from src.a_extract_pose import extract_keypoints_from_video
from src.b_compute_motion_signal import compute_motion_signal
from src.c_reconstruction import  synchronize_keypoints,triangulate_pose, visualize_3d_pose, P1, P2
import mediapipe as mp

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def visualize_motion_graph(npy_path, title="Motion Signal"):
    """
    Hiển thị đồ thị tín hiệu chuyển động từ file npy keypoints
    """
    import matplotlib.pyplot as plt
    
    keypoints = np.load(npy_path)
    
    velocity = np.diff(keypoints, axis=0)
    speed_per_joint = np.linalg.norm(velocity, axis=2)
    motion_signal = np.mean(speed_per_joint, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(motion_signal)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Average Speed (normalized)")
    plt.grid(True)
    plt.show()


def visualize_keypoints(video_path, result_folder="result", result_video_folder="result_video", npy_folder=None):
    """
    Trích xuất keypoints và vẽ lên video, lưu vào thư mục tương ứng
    """
    if npy_folder is None:
        npy_folder = result_folder
        
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    
    npy_path = os.path.join(npy_folder, f"{name_without_ext}_keypoints.npy")
    output_video_path = os.path.join(result_video_folder, f"{name_without_ext}_with_keypoints.mp4")
    
    if os.path.exists(npy_path):
        print(f"Đang tải keypoints từ: {npy_path}")
        keypoints_array = np.load(npy_path)
    else:
        print("Đang trích xuất keypoints...")
        keypoints_array = extract_keypoints_from_video(video_path, output_npy=npy_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video!")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    print(f"Đang vẽ keypoints và lưu video vào: {output_video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(keypoints_array):
            pose_landmarks = keypoints_array[frame_idx]
            
            if np.any(pose_landmarks != 0):
                points = [(int(x * width), int(y * height)) for x, y in pose_landmarks]
                
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(points) and end_idx < len(points):
                        cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 255), 2)
                
                for point in points:
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"Đã xử lý {frame_idx}/{len(keypoints_array)} frames")
    
    cap.release()
    out.release()
    
    print(f"Hoàn thành!")
    print(f"   • Keypoints: {npy_path}")
    print(f"   • Video:     {output_video_path}")


# ====================== CHẠY ======================
if __name__ == "__main__":
    vids_dir = "vids"
    
    if not os.path.exists(vids_dir):
        print(f"Thư mục {vids_dir} không tồn tại!")
        exit(1)
    
    # Lấy tất cả subfolders
    subfolders = [f for f in os.listdir(vids_dir) if os.path.isdir(os.path.join(vids_dir, f))]
    
    if not subfolders:
        print(f"Không tìm thấy subfolder nào trong {vids_dir}")
        exit(1)
    
    print(f"Tìm thấy {len(subfolders)} subfolder(s): {subfolders}")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(vids_dir, subfolder)
        result_subfolder = os.path.join("result", subfolder)
        result_video_subfolder = os.path.join("result_video", subfolder)
        npy_subfolder = os.path.join(result_subfolder, "npy")
        motion_subfolder = os.path.join(result_subfolder, "motion")
        
        os.makedirs(result_subfolder, exist_ok=True)
        os.makedirs(result_video_subfolder, exist_ok=True)
        os.makedirs(npy_subfolder, exist_ok=True)
        os.makedirs(motion_subfolder, exist_ok=True)
        
        # Lấy videos trong subfolder
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = [f for f in os.listdir(subfolder_path) if os.path.splitext(f)[1].lower() in video_extensions]
        
        if not video_files:
            print(f"Không tìm thấy video trong {subfolder}")
            continue
        
        print(f"\nXử lý subfolder: {subfolder} ({len(video_files)} videos)")
        for vf in video_files:
            print(f"   • {vf}")
        
        for video_file in video_files:
            video_path = os.path.join(subfolder_path, video_file)
            print(f"  Đang xử lý: {video_file}")
            visualize_keypoints(video_path, result_subfolder, result_video_subfolder, npy_subfolder)
            
            video_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(video_name)[0]
            npy_path = os.path.join(npy_subfolder, f"{name_without_ext}_keypoints.npy")
            plot_path = os.path.join(motion_subfolder, f"{name_without_ext}_motion_signal.png")
            compute_motion_signal(npy_path, plot_path)
    
    print("\n[OK] Processed all subfolders!")


    kp1 = np.load("result/ballet/npy/BalletDance_1_keypoints.npy")
    kp2 = np.load("result/ballet/npy/BalletDance_2_keypoints.npy")
    kp1_sync, kp2_sync = synchronize_keypoints(kp1, kp2)

    

    # # Triangulate 3D poses for all frames
    # points_3d_all = triangulate_pose(kp1, kp2, P1, P2)  
    # # Visualize all 3D poses as an animation
    # visualize_3d_pose(points_3d_all)
