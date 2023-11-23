import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets
import random
#pip install opencv-python mediapip PyQt5

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class VideoStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(VideoStreamWidget, self).__init__(parent)
        self.video_frame = QtWidgets.QLabel()
        self.roi_frame = QtWidgets.QLabel()

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.video_frame)
        layout.addWidget(self.roi_frame)
        self.setLayout(layout)

        self.capture = cv2.VideoCapture(0)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

        self.old_gray = None
        self.p0 = None  # Initial points to track

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def update(self):
        ret, frame = self.capture.read()
        if ret:
            processed_frame, roi, self.old_gray, self.p0 = self.process_frame(frame, self.old_gray, self.p0)
            image = self.convert_frame(processed_frame)
            self.video_frame.setPixmap(QtGui.QPixmap.fromImage(image))

            if roi is not None:
                roi_image = self.convert_frame(roi)
                self.roi_frame.setPixmap(QtGui.QPixmap.fromImage(roi_image))

    def convert_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format

    def process_frame(self, frame, old_gray, p0):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.pose.process(frame_rgb)
        roi = None

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            x_min = int(min(left_shoulder.x, right_shoulder.x) * frame.shape[1])
            x_max = int(max(left_shoulder.x, right_shoulder.x) * frame.shape[1])
            y_min = int(min(left_shoulder.y, right_shoulder.y) * frame.shape[0])
            y_max = int(max(left_hip.y, right_hip.y) * frame.shape[0])

            if x_min < x_max and y_min < y_max and x_max - x_min > 0 and y_max - y_min > 0:
                roi = frame[y_min:y_max, x_min:x_max].copy()

                if roi.size == 0:
                    return frame, None, old_gray, p0

                if old_gray is None or p0 is None:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=25, qualityLevel=0.01, minDistance=7)
                    if p0 is not None:
                        p0 = p0 + np.array([x_min, y_min])
                else:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
                    if p1 is not None and st is not None:
                        for i, (new, _) in enumerate(zip(p1, p0)):
                            if st[i]:
                                a, b = new.ravel()
                                cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

                    p0 = p1

        return frame, roi, frame_gray, p0

def main():
    app = QtWidgets.QApplication(sys.argv)
