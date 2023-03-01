import cv2
import numpy as np

def retrieve_odometry(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features in the first frame
    features1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Track features in the second frame using optical flow
    features2, status, errors = cv2.calcOpticalFlowPyrLK(gray1, gray2, features1, None)

    # Select only the features that were successfully tracked
    good1 = features1[status == 1]
    good2 = features2[status == 1]

    # Compute the essential matrix from the corresponding feature points
    E, mask = cv2.findEssentialMat(good1, good2, focal=1.0, pp=(0, 0))

    # Recover the relative camera motion from the essential matrix
    _, rotation, translation, mask = cv2.recoverPose(E, good1, good2, focal=1.0, pp=(0, 0))

    return rotation, translation


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[INF] Failed to open pipeline ...")
    exit()
else:
    success, img1 = cap.read()

# Load two consecutive frames
while cap.isOpened():
    success, img2 = cap.read()

    R, t = retrieve_odometry(img1, img2)

    # Print the rotation and translation vectors
    print("Rotation vector:\n", R)
    print("Translation vector:\n", t)

    cv2.imshow("image", img2)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    img1 = img2

cap.release()
cv2.destroyAllWindows()