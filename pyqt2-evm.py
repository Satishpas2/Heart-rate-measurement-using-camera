import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import mediapipe as mp

class VideoStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(VideoStreamWidget, self).__init__(parent)
        self.video_frame = QtWidgets.QLabel()
        self.roi_frame = QtWidgets.QLabel()  # Separate QLabel for displaying ROI
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_frame)
        layout.addWidget(self.roi_frame)  # Add the ROI QLabel to the layout
        self.setLayout(layout)

        self.capture = cv2.VideoCapture(0)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

        # Timer for updating the video frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # Update every 30 milliseconds

    def update(self):
        ret, frame = self.capture.read()
        if ret:
            frame = self.process_frame(frame)
            image = self.convert_frame(frame)
            self.video_frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def convert_frame(self, frame):
        """ Convert the image format for PyQt compatibility. """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format

    def process_frame(self, frame):
        """ Process frame to add pose estimation using MediaPipe and track feature points """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Get shoulder and hip points
            shoulder_left = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            hip_left = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            hip_right = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            # Calculate ROI coordinates
            roi_x = int(min(shoulder_left.x, shoulder_right.x) * frame.shape[1])
            roi_y = int(shoulder_left.y * frame.shape[0])
            roi_width = int(abs(shoulder_right.x - shoulder_left.x) * frame.shape[1])
            roi_height = int(abs(hip_right.y - shoulder_right.y) * frame.shape[0])

            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

            # Get feature points within ROI
            roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            feature_points = cv2.goodFeaturesToTrack(roi_gray, 25, 0.01, 10)
            if feature_points is not None:
                feature_points = np.int0(feature_points)
                for point in feature_points:
                    x, y = point.ravel()
                    cv2.circle(roi_frame, (x, y), 5, (0, 0, 255), -1)

            # Update frame with ROI and feature points
            frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = roi_frame

            # Convert the ROI frame for display
            roi_image = self.convert_frame(roi_frame)
            self.roi_frame.setPixmap(QtGui.QPixmap.fromImage(roi_image))
            self.roi_frame.setText("ROI")  # Set label text as "ROI"

            # Draw ROI rectangle on the main frame
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

        self.video_frame.setText("Pose Detection")  # Set label text as "Pose Detection"
        return frame

def main():
    app = QtWidgets.QApplication(sys.argv)
    video_stream_widget = VideoStreamWidget()
    video_stream_widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
