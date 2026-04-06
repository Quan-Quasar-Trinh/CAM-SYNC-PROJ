import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_keypoints_from_video(video_path, output_npy="keypoints.npy"):
    # Khởi tạo Pose Landmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False
    )
    
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video!")
        return
    
    keypoints_list = []
    frame_idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc="Extracting keypoints") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Tính timestamp
            timestamp_ms = int(frame_idx * 1000 / cap.get(cv2.CAP_PROP_FPS))
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.pose_landmarks:
                # Lấy 33 keypoints (x, y) normalized
                frame_kp = np.array([[lm.x, lm.y] for lm in result.pose_landmarks[0]])
            else:
                # Nếu không detect được thì dùng zeros
                frame_kp = np.zeros((33, 2))
            
            keypoints_list.append(frame_kp)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    landmarker.close()
    
    keypoints_array = np.array(keypoints_list)  # shape: (num_frames, 33, 2)
    np.save(output_npy, keypoints_array)
    print(f"Đã lưu {keypoints_array.shape} keypoints vào {output_npy}")
    
    return keypoints_array

# Sử dụng
# extract_keypoints_from_video("cam1.mp4", "cam1_keypoints.npy")
# extract_keypoints_from_video("cam2.mp4", "cam2_keypoints.npy")