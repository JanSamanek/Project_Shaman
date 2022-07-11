from holistic_detector import *

model = load_model('mp_hand_gesture')

with open('gesture.names', 'r') as file:
    class_names = file.read().split('\n')
print(class_names)

vision_cap = cv2.VideoCapture(0)    # TODO remove inbuilt camera view
detector = Detector()
previous_time = 0

while True:
    # reading the image from video capture
    _, img = vision_cap.read()
    img = detector.init_landmarks(img)

    landmarks = detector.get_landmarks(img, 'right_hand')
    # previous_time = display_fps(img, previous_time)

    # Predict gesture in Hand Gesture Recognition project
    try:
        prediction = model.predict([landmarks])
    except KeyError:
        print('No hand to evaluate')
    else:
        print(prediction)
        classID = np.argmax(prediction)
        className = class_names[classID]
        cv2.putText(img, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Vision", img)
    cv2.waitKey(1)
