import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

def get_roi_from_pose(frame):
    """Detect pose and return coordinates for the ROI (between shoulders and hips)."""
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    shoulder1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    shoulder2 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    hip1 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip2 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    frame_height, frame_width, _ = frame.shape
    x_min = int(min(shoulder1.x, shoulder2.x, hip1.x, hip2.x) * frame_width)
    x_max = int(max(shoulder1.x, shoulder2.x, hip1.x, hip2.x) * frame_width)
    y_min = int(min(shoulder1.y, shoulder2.y, hip1.y, hip2.y) * frame_height)
    y_max = int(max(shoulder1.y, shoulder2.y, hip1.y, hip2.y) * frame_height)

    return x_min, y_min, x_max - x_min, y_max - y_min

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow between two frames."""
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def calculate_breathing_rate(breathing_signals, fps):
    """Calculate the breathing rate from the breathing signals."""
    peaks, _ = find_peaks(breathing_signals)
    breathing_rate = len(peaks) * (60 / fps)
    return breathing_rate

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    breathing_signals = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = get_roi_from_pose(frame)
        if roi:
            x, y, w, h = roi
            curr_roi = curr_gray[y:y+h, x:x+w]
            prev_roi = prev_gray[y:y+h, x:x+w]

            flow = calculate_optical_flow(prev_roi, curr_roi)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            breathing_signal = np.mean(magnitude)
            breathing_signals.append(breathing_signal)

            # Draw the ROI for visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Breathing Rate Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray

    cap.release()
    cv2.destroyAllWindows()

    # Calculate breathing rate
    breathing_signals = np.array(breathing_signals)
    rate = calculate_breathing_rate(breathing_signals, fps)
    print(f"Estimated Breathing Rate: {rate:.2f} breaths per minute")

if __name__ == "__main__":
    video_path = '/media/satish/data/data-openpose/99a1ae31-b589-4da8-97c8-752090123444.mp4'  # Replace with your video path
    main(video_path)
