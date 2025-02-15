import numpy as np 
import cv2 

cap = cv2.VideoCapture('/media/satish/data/data-openpose/video/pexels-nicky-pe-13369724 (Original).mp4')

# params for corner detection 
feature_params = dict( maxCorners = 100, 
                    qualityLevel = 0.3, 
                    minDistance = 7, 
                    blockSize = 7 ) 

# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (15, 15), 
                maxLevel = 2, 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                            10, 0.03)) 

# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 

# Take first frame and find corners in it 
ret, old_frame = cap.read() 
old_gray = cv2.cvtColor(old_frame, 
                        cv2.COLOR_BGR2GRAY) 

# Select ROI
r = cv2.selectROI(old_frame)
cv2.destroyAllWindows()

# Crop ROI from the frame
roi = old_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Find corners in the ROI
p0 = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params) 

# Create a mask image for drawing purposes 
mask = np.zeros_like(old_frame) 

while(1): 
    
    ret, frame = cap.read() 
    frame_gray = cv2.cvtColor(frame, 
                            cv2.COLOR_BGR2GRAY) 

    # calculate optical flow 
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, 
                                        frame_gray, 
                                        p0, None, 
                                        **lk_params) 

    # Select good points 
    good_new = p1[st == 1] 
    good_old = p0[st == 1] 

    # draw the tracks 
    for i, (new, old) in enumerate(zip(good_new, 
                                    good_old)): 
        a, b = new.ravel() 
        c, d = old.ravel() 
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                        color[i].tolist(), 2) 
        
        frame = cv2.circle(frame, (int(a), int(b)), 5, 
                        color[i].tolist(), -1) 
        
    img = cv2.add(frame, mask) 

    cv2.imshow('frame', img) 
    
    k = cv2.waitKey(25) 
    if k == 27: 
        break

    # Updating Previous frame and points 
    old_gray = frame_gray.copy() 
    p0 = good_new.reshape(-1, 1, 2) 

cv2.destroyAllWindows() 
cap.release() 
