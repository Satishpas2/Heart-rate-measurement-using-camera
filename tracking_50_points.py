import cv2
import numpy as np
import mediapipe as mp

def select_equidistant_points(p1, p2, num_points):
    """ Select num_points equidistant points between p1 and p2 """
    return np.linspace(p1, p2, num_points)

def track_breathing_rate(frame):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Detect pose
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None

    # Get landmarks
    landmarks = results.pose_landmarks.landmark
    shoulders = np.array([[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                          [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]])
    hips = np.array([[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                     [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]])

    # Convert to pixel coordinates
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    shoulders = (shoulders * [frame_width, frame_height]).astype(int)
    hips = (hips * [frame_width, frame_height]).astype(int)

    # Define ROI
    roi_corners = np.array([shoulders[0], shoulders[1], hips[1], hips[0]])

    # Select 50 equidistant points within the ROI
    top_line = select_equidistant_points(roi_corners[0], roi_corners[1], 10)
    bottom_line = select_equidistant_points(roi_corners[3], roi_corners[2], 10)
    left_line = select_equidistant_points(roi_corners[0], roi_corners[3], 15)
    right_line = select_equidistant_points(roi_corners[1], roi_corners[2], 15)
    points = np.concatenate((top_line, bottom_line, left_line, right_line)).astype(np.float32)

    # Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    p0 = points.reshape(-1, 1, 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
        
        img = cv2.add(frame, mask)

        cv2.imshow('Tracking', img)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Example usage
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    track_breathing_rate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
cap.release()
