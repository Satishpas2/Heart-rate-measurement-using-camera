import cv2
import numpy as np
import eulerian_magnification as em

def process_frame(frame, fps, freq_min=50.0 / 60.0, freq_max=1.0, amplification=50, pyramid_levels=3):
    # Apply Eulerian Magnification to a single frame
    # Note: This function might need to be adjusted based on how em.eulerian_magnification processes frames
    magnified_frame = em.eulerian_magnification(
        frame, fps, freq_min=freq_min, freq_max=freq_max, amplification=amplification, pyramid_levels=pyramid_levels
    )
    return magnified_frame

def main():
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)

    # Assuming a common webcam FPS
    fps = 30 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        magnified_frame = process_frame(frame, fps)

        # Display the magnified frame
        cv2.imshow('Eulerian Magnified Video', magnified_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
