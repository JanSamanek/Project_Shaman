import cv2 as cv
import time


def display_fps(img, previous_time):
    # measuring and displaying fps
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    return previous_time


pipeline = "gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
# print(cv.getBuildInformation())

cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open pipeline")
    exit()
    
previous_time = 0
while cap.isOpened():
    success, img = cap.read()
    
    previous_time = display_fps(img, previous_time)
    
    if not success:
        print('empty frame')
        continue
    
    cv.imshow("GSTREAMER", img)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink
# gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host=10.0.1.103 port=5000