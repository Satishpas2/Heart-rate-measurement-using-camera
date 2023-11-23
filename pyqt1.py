import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

class VideoStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(VideoStreamWidget, self).__init__(parent)

        self.video_frame = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_frame)
        self.setLayout(layout)

        self.capture = cv2.VideoCapture(0)

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
        """ Process frame to convert it to grayscale """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

def main():
    app = QtWidgets.QApplication(sys.argv)
    video_stream_widget = VideoStreamWidget()
    video_stream_widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
