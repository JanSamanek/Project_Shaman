import cv2 as cv

pipeline = "udpsrc port=8000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
print(cv.getBuildInformation())

cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open pipeline")
    exit()
    
while True:
    success, img = cap.read()
    
    if not success:
        print('empty frame')
        continue
    
    cv.imshow("GSTREAMER", img)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()