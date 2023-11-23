import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import find_peaks

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def select_equidistant_points(start_point, end_point, num_points):
    return np.linspace(start=start_point, stop=end_point, num=num_points)

def get_roi_landmarks(landmarks):
    return np.array([(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                     (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                     (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                     (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)])

def apply_magnification(frame, intensity=2):
    # Simple magnification effect by amplifying deviations from the mean
    frame_mean = np.mean(frame, axis=(0, 1))
    magnified = intensity * (frame - frame_mean) + frame_mean
    return np.clip(magnified, 0, 255).astype(np.uint8)

def calculate_breathing_rate(breathing_signals, fps):
    peaks, _ = find_peaks(breathing_signals, height=0.1, distance=fps/2)
    breathing_rate = len(peaks) * (60 / fps)
    return breathing_rate

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    breathing_signals = []
    tracking_points = None

    cv2.namedWindow('Original')
    cv2.namedWindow('Tracking')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply video magnification
        magnified_frame = apply_magnification(frame)
        
        # Display the original and magnified video
        cv2.imshow('Original', frame)
        cv2.imshow('Magnified', magnified_frame)

        # MediaPipe pose detection
        results = pose.process(cv2.cvtColor(magnified_frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            roi_landmarks = get_roi_landmarks(landmarks)
            frame_width, frame_height = frame.shape[1], frame.shape[0]

            if tracking_points is None:
                # Initialize tracking points within the ROI
                tracking_points = []
                for i in range(25):
                    start_point = np.array([roi_landmarks[0][0] + (roi_landmarks[2][0] - roi_landmarks[0][0]) * i / 24,
                                            roi_landmarks[0][1] + (roi_landmarks[2][1] - roi_landmarks[0][1]) * i / 24])
                    end_point = np.array([roi_landmarks[1][0] + (roi_landmarks[3][0] - roi_landmarks[1][0]) * i / 24,
                                          roi_landmarks[1][1] + (roi_landmarks[3][1] - roi_landmarks[1][1]) * i / 24])
                    points_line = select_equidistant_points(start_point, end_point, 2)
                    tracking_points.extend(points_line)
                tracking_points = np.array(tracking_points, dtype=np.float32).reshape(-1, 2)  # Fix the shape of tracking_points

            curr_gray = cv2.cvtColor(magnified_frame, cv2.COLOR_BGR2GRAY)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, tracking_points, None)

            if next_points is not None and status is not None:
                good_points = next_points[status.flatten() == 1]
                for pt in good_points:
                    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 3, (0, 0, 255), -1)
                tracking_points = good_points.reshape(-1, 1, 2)
                displacements = good_points[:, 1] - tracking_points[:, 0, 1]
                breathing_signals.append(np.mean(displacements))

            prev_gray = curr_gray

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    breathing_rate = calculate_breathing_rate(np.array(breathing_signals), fps)
    print(f"Estimated Breathing Rate: {breathing_rate:.2f} breaths per minute")

if __name__ == "__main__":
    video_path = '/media/satish/data/data-openpose/99a1ae31-b589-4da8-97c8-752090123444.mp4'  # Replace with your video path
    main(video_path)
