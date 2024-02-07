## code for optical flow using Lukas-kanade method mediapipe, opencv,

import numpy as np
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(2)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize Mediapipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Initialize variables for plotting
    plot_x = []
    plot_y = []
    initial_landmarks = None

    while (1):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect pose landmarks
        results = pose.process(frame)
        if results.pose_landmarks:
            # Get landmarks for shoulders and hips
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            # Convert landmarks to pixel coordinates
            p0 = np.array([[left_shoulder.x * frame.shape[1], left_shoulder.y * frame.shape[0]],
                           [right_shoulder.x * frame.shape[1], right_shoulder.y * frame.shape[0]],
                           [left_hip.x * frame.shape[1], left_hip.y * frame.shape[0]],
                           [right_hip.x * frame.shape[1], right_hip.y * frame.shape[0]]], dtype=np.float32)

            # Create a rectangle using the landmarks
            x, y, w, h = cv2.boundingRect(p0)
            rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the step size for selecting equidistant landmarks
            step_size = min(w, h) // 5

            # Select 25 equidistant landmarks inside the rectangle
            equidistant_landmarks = []
            for i in range(5):
                for j in range(5):
                    landmark_x = x + (i * step_size)
                    landmark_y = y + (j * step_size)
                    equidistant_landmarks.append([landmark_x, landmark_y])

            # Convert the equidistant landmarks to numpy array
            equidistant_landmarks = np.array(equidistant_landmarks, dtype=np.float32)
            p0 = equidistant_landmarks
            #print(type(p0))
            #print(type(equidistant_landmarks))
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st[:, 0] == 1]  # Fix the indexing error

            good_old = p0[st[:, 0] == 1].reshape(-1, 1, 2)  # Fix the indexing error

            # Create a separate image for displaying the changes in landmarks
            landmark_changes = np.zeros_like(frame)

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                
                # Draw the changes in landmarks on the separate image
                landmark_changes = cv2.line(landmark_changes, (int(c), int(d)), (int(a), int(b)), color[i].tolist(), 2)
                landmark_changes = cv2.circle(landmark_changes, (int(a), int(b)), 5, color[i].tolist(), -1)

                # Plot the changes in a single landmark
                if i == 0:
                    plot_x.append(len(plot_x))
                    plot_y.append(int(b) - int(d))

            # Combine the original frame and the changes in landmarks image
            combined_frame = np.hstack((frame, landmark_changes))

            # Display the combined frame
            cv2.imshow('frame', combined_frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # Calculate the averaged initial value of all landmarks in the first frame
            if initial_landmarks is None:
                initial_landmarks = np.mean(p0, axis=0)

            # Calculate the changes in averaged value of all 25 landmarks
            averaged_landmarks = np.mean(p0, axis=0)
            averaged_changes = averaged_landmarks - initial_landmarks

            # Plot the changes in averaged value over time
            plot_frame = np.zeros((600, 1600, 3), dtype=np.uint8)
            cv2.line(plot_frame, (0, 300), (1600, 300), (255, 255, 255), 1)  # X-axis
            cv2.line(plot_frame, (50, 0), (50, 600), (255, 255, 255), 1)  # Y-axis

            # Plot the data points
            #for i in range(len(plot_x)):
            cv2.circle(plot_frame, (50 , 300 - int(averaged_changes[0][1] * 3)), 5, (0, 0, 255), -1)
            print("averaged_changes",int(averaged_changes[0][1] * 2))
            #print(initial_landmarks)

            # Display the plot
            cv2.imshow('plot-Avg-change', plot_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()
cap.release()