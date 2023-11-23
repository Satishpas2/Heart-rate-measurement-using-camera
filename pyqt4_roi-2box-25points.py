import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import mediapipe as mp

class VideoStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(VideoStreamWidget, self).__init__(parent)

        # Initialize the labels for full frame and ROI
        self.video_frame = QtWidgets.QLabel()
        self.roi_frame = QtWidgets.QLabel()

        # Layout for the labels
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.video_frame)
        layout.addWidget(self.roi_frame)
        self.setLayout(layout)

        self.capture = cv2.VideoCapture("/media/satish/data/data-openpose/rgb1-evm-magnified.mp4") ##############    Change camera number here

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

        # Timer for updating the video frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # Update every 30 milliseconds

    def start_webcam_stream(self):
        """ Start streaming video from the webcam """
        self.capture = cv2.VideoCapture(0)  # Change camera number here

    def start_file_stream(self, filepath):
        """ Start streaming video from a saved file """
        self.capture = cv2.VideoCapture(filepath)

    def update(self):
        ret, frame = self.capture.read()
        if ret:
            result = self.process_frame(frame)
            if result is not None:
                processed_frame, roi = result
                image = self.convert_frame(processed_frame)
                self.video_frame.setPixmap(QtGui.QPixmap.fromImage(image))

                if roi is not None:
                    roi_image = self.convert_frame(roi)
                    self.roi_frame.setPixmap(QtGui.QPixmap.fromImage(roi_image))

    def convert_frame(self, frame):
        """ Convert the image format for PyQt compatibility. """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format

   

    def process_frame(self, frame):
        """ Process frame to add pose estimation, extract ROI, and track points """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        roi = None
        points = None

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Get landmark positions
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

            # Calculate ROI coordinates
            x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * frame.shape[1])
            x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * frame.shape[1])
            y_min = int(min(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y) * frame.shape[0])
            y_max = int(max(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y) * frame.shape[0])

            # Add a margin to the ROI
            margin = 10  # pixels
            x_min = max(0, x_min + margin)
            x_max = min(frame.shape[1], x_max - margin)
            y_min = max(0, y_min + margin)
            y_max = min(frame.shape[0], y_max - margin)

            # Extract ROI
            if x_min < x_max and y_min < y_max:
                roi = frame[y_min:y_max, x_min:x_max]

            # Convert ROI to grayscale
            gray_roi = None
            if roi is not None:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Find good features to track
            points = None
            if gray_roi is not None:
                points = cv2.goodFeaturesToTrack(gray_roi, maxCorners=25, qualityLevel=0.01, minDistance=10)

            # Convert points to their original coordinates
            if points is not None:
                points = np.int0(points)
                for i in points:
                    x, y = i.ravel()
                    cv2.circle(frame, (x_min + x, y_min + y), 3, 255, -1)

            return frame, roi

def main():
    app = QtWidgets.QApplication(sys.argv)
    video_stream_widget = VideoStreamWidget()
    video_stream_widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
