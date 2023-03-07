import cv2
import numpy as np
from Utilities.display_functions import display_fps

def get_camera_shift(img1, img2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match the keypoints in the two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Find the homography matrix that maps img1 to img2
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Extract the translation component of the homography matrix
    dx = M[0, 2]

    return dx

def tracker_test():
    # Read video file
    cap = cv2.VideoCapture(0)

    # Define the tracker
    tracker = cv2.Tracker_create()

    # Read the first frame and select a point to track
    ret, frame = cap.read()
    bbox = cv2.selectROI(frame, False)
    tracker.init(frame, bbox)

    # Loop through the video frames and track the selected point
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker with the current frame
        success, bbox = tracker.update(frame)

        # If the tracking was successful, draw a rectangle around the tracked point
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the current frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

def shift_test():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[INF] Failed to open pipeline ...")
        exit()
    else:
        success, img1 = cap.read()

    previous_time = 0

    # Load two consecutive frames
    while cap.isOpened():
        success, img2 = cap.read()

        dx = get_camera_shift(img1, img2)

        previous_time = display_fps(img2, previous_time)
        
        if dx < 0:
            print("LEFT")
        else:
            print("RIGHT")
            
        cv2.imshow("image", img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img1 = img2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    shift_test()
