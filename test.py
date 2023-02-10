from jetcam.csi_camera import CSICamera
import cv2

camera = CSICamera(width=300, height=300)

camera.running = True

def execute(change):
    img = change['new']
    cv2.imshow("hello", img)
    cv2.waitKey(1)
    
camera.observe(execute, names='value')