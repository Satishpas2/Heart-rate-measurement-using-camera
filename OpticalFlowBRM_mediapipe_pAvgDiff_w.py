import os
import numpy as np
import cv2
import time
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0
y_prev = 0

cap = cv2.VideoCapture(2)

# specify the file path
file_path = 'p_diff_avg.txt'
# check if file exists
if os.path.isfile(file_path):
    # remove the file
    os.remove(file_path)
else:
    print("Error: %s file not found" % file_path)            

while True:

    # start time to calculate FPS
    start = time.time()

    suc, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    # wait for 3 seconds for next frameabs(p0-p0r).reshape(-1, 2).max(-1)abs(p0-p0r).reshape(-1, 2).max(-1)abs(p0-p0r).reshape(-1, 2).max(-1)abs(p0-p0r).reshape(-1, 2).max(-1)
    #time.sleep(3)
    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []
        
        p_diff_avg=0
        p_diff = p1-p0
        for i in p_diff:
            i[0][1]
            p_diff_avg=p_diff_avg+i[0][1]
            p_diff_avg=p_diff_avg     ####/len(p_diff)
        print("p_diff_avg",p_diff_avg)
        with open('p_diff_avg.txt', 'a') as f:
                f.write(str(p_diff_avg  ))
                # f.write("  prev  :"+str(y_prev  ))
                # f.write("  diff  :"+str(y_diff  ))
                f.write('\n')
        #y_ref=415
        # y_avg = 0
        # count=0
        
        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            # count+=1
            # y_avg = (y_avg +y)/count
            # y_diff = y_avg - y_prev
            # print("count",count)
            # print("y_diff",y_avg)
            # print("y_diff",y_prev)
            # print("y_diff",y_diff)
            #print("frame_idx",frame_idx)
            #Save y_diff to a file
            
            # with open('p_diff_avg.txt', 'a') as f:
            #     f.write("  avg  :"+str(p_diff_avg  ))
            #     # f.write("  prev  :"+str(y_prev  ))
            #     # f.write("  diff  :"+str(y_diff  ))
            #     f.write('\n')
               
            #y_prev = y_avg
            #print(y_avg)

        trajectories = new_trajectories
        # print(len(trajectories))
        # print(len(trajectory))
        # Draw all the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)


    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        #mask = np.zeros_like(frame_gray)
        #mask[int(frame_gray.shape[0]/4):int(frame_gray.shape[0]*3/4), int(frame_gray.shape[1]/4):int(frame_gray.shape[1]*3/4)] = 255
        results = pose.process(frame)
        landmarks = results.pose_landmarks
        if landmarks is not None:
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
                mask = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
        else:
            continue
            # mask = np.zeros_like(frame_gray)
            # mask[int(frame_gray.shape[0]/4):int(frame_gray.shape[0]*3/4), int(frame_gray.shape[1]/4):int(frame_gray.shape[1]*3/4)] = 255
 
                    
        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])


    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    
    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
