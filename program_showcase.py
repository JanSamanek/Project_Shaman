from tracker import create_tracker
from Pose.pose_detector import PoseDetector
import cv2 as cv
from Utilities.display import display_fps

def main():
    cap = cv.VideoCapture(0)
    previous_time = 0
    tracker = None
    pose_detector = PoseDetector()
    pose_img = None

    while cap.isOpened():
        success, img = cap.read()

        if tracker is not None:
            pose_img = img.copy()
            img = tracker.track(img)
            to_box = tracker.tracked_to.box if tracker.tracked_to is not None else None

            if to_box is not None:
                pose_img = pose_detector._get_landmarks(pose_img, box=to_box)

                # has to be called after get landmarks
                if pose_detector._detect_left_hand_above_nose() and pose_detector._detect_right_hand_above_nose():
                    print("[INF] Both hands above nose detected")
                elif pose_detector._detect_left_hand_above_nose():
                    print("[INF] Left hand gesture above nose detected")
                elif pose_detector._detect_right_hand_above_nose():
                    print("[INF] Right hand gesture above nose detected")
            else:
                pose_img = None

        previous_time = display_fps(img, previous_time)

        cv.imshow('detector', img)
        if pose_img is not None:
            cv.imshow('pose', pose_img)

        if cv.waitKey(1) & 0xFF == ord('s'):
            tracker = create_tracker(img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
     main()


#mot_speed_1 = speed + turn_gain * center_x
#mot_speed_2 = speed - turn_gain * center_x
# max distance important!
# sudo apt-get install mosquitto mosquitto-clients
# pip install paho-mqtt

# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink
# gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host=192.168.0.159 port=5000

# git push https://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@github.com/JanSamanek/Project_Shaman.git
# token: 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