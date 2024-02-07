import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame = cv2.imread('/home/satish/Downloads/bhonu_avatar/image (3).png')
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

results = pose.process(frame)
landmarks = results.pose_landmarks
LEFT_SHOULDER =landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
RIGHT_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
LEFT_HIP = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
RIGHT_HIP = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
# shoulder_landmarks = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
image_height, image_width, _ = frame.shape

shoulder_landmarks = [
    (int(LEFT_SHOULDER.x * image_width), int(LEFT_SHOULDER.y * image_height)),
    (int(RIGHT_SHOULDER.x * image_width), int(RIGHT_SHOULDER.y * image_height))
]

hip_landmarks = [
    (int(LEFT_HIP.x * image_width), int(LEFT_HIP.y * image_height)),
    (int(RIGHT_HIP.x * image_width), int(RIGHT_HIP.y * image_height))
]

# Create a mask for good features to track
#mask = np.zeros_like(frame_gray)
#cv2.fillPoly(mask, [np.array(shoulder_landmarks + hip_landmarks)], 255)
mask = np.zeros_like(frame_gray)
x1, y1 = shoulder_landmarks[0]
x2, y2 = shoulder_landmarks[1]
x3, y3 = hip_landmarks[0]
x4, y4 = hip_landmarks[1]
cv2.rectangle(mask, (x4, y4), (x1, y1), 255, -1)
#cv2.rectangle(mask, (x3, y3), (x4, y4), 255, -1)
# Apply the mask to the grayscale image
masked_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

# Apply the good features to track algorithm
corners = cv2.goodFeaturesToTrack(masked_gray, maxCorners=200, qualityLevel=0.1,
                                  minDistance=10, blockSize=7)

# Display the image with good features
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

cv2.imshow("Image with Good Features", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
