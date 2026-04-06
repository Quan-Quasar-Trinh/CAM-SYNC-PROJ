import cv2
import numpy as np
import os
from src.a import extract_keypoints_from_video
from src.b import compute_motion_signal
import mediapipe as mp

# ================== TẠO THƯ MỤC ==================
RESULT_FOLDER = "result"
RESULT_VIDEO_FOLDER = "result_video"

os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(RESULT_VIDEO_FOLDER, exist_ok=True)

# Định nghĩa các kết nối cho pose
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
    
    # Tính velocity
    velocity = np.diff(keypoints, axis=0)
    speed_per_joint = np.linalg.norm(velocity, axis=2)
    motion_signal = np.mean(speed_per_joint, axis=1)
    
    # Vẽ
    plt.figure(figsize=(12, 6))
    plt.plot(motion_signal)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Average Speed (normalized)")
    plt.grid(True)
    plt.show()


def visualize_keypoints(video_path):
    """
    Trích xuất keypoints và vẽ lên video, lưu vào thư mục tương ứng
    """
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    
    # Đường dẫn lưu file
    npy_path = os.path.join(RESULT_FOLDER, f"{name_without_ext}_keypoints.npy")
    output_video_path = os.path.join(RESULT_VIDEO_FOLDER, f"{name_without_ext}_with_keypoints.mp4")
    
    # 1. Kiểm tra xem keypoints đã có chưa
    if os.path.exists(npy_path):
        print(f"Đang tải keypoints từ: {npy_path}")
        keypoints_array = np.load(npy_path)
    else:
        # Trích xuất keypoints nếu chưa có
        print("Đang trích xuất keypoints...")
        keypoints_array = extract_keypoints_from_video(video_path, output_npy=npy_path)
    
    # 2. Đọc video gốc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video!")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo VideoWriter
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
            
            if np.any(pose_landmarks != 0):  # Không phải frame trống
                # Convert normalized coordinates to pixel coordinates
                points = [(int(x * width), int(y * height)) for x, y in pose_landmarks]
                
                # Draw connections
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(points) and end_idx < len(points):
                        cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 255), 2)
                
                # Draw keypoints
                for point in points:
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)
        
        out.write(frame)
        frame_idx += 1
        
        # Hiển thị tiến trình
        if frame_idx % 50 == 0:
            print(f"Đã xử lý {frame_idx}/{len(keypoints_array)} frames")
    
    cap.release()
    out.release()
    
    print(f"✅ Hoàn thành!")
    print(f"   • Keypoints: {npy_path}")
    print(f"   • Video:     {output_video_path}")


# ====================== CHẠY ======================
if __name__ == "__main__":
    test_dir = "vids\\test"
    
    if not os.path.exists(test_dir):
        print(f"❌ Thư mục {test_dir} không tồn tại!")
        exit(1)
    
    # Lấy tất cả file video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = [f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"❌ Không tìm thấy file video nào trong {test_dir}")
        exit(1)
    
    print(f"📁 Tìm thấy {len(video_files)} video(s) để xử lý:")
    for vf in video_files:
        print(f"   • {vf}")
    
    for video_file in video_files:
        video_path = os.path.join(test_dir, video_file)
        print(f"\n🔄 Đang xử lý: {video_file}")
        visualize_keypoints(video_path)
        
        # Tính tín hiệu chuyển động
        video_name = os.path.basename(video_path)
        name_without_ext = os.path.splitext(video_name)[0]
        npy_path = os.path.join(RESULT_FOLDER, f"{name_without_ext}_keypoints.npy")
        plot_path = os.path.join(RESULT_FOLDER, f"{name_without_ext}_motion_signal.png")
        compute_motion_signal(npy_path, plot_path)
    
    print("\n✅ Đã xử lý tất cả video!")