from yolov5 import train

if __name__ == '__main__':
    train.run(imgsz=640, data='dataset.yaml', weights='yolov5s.pt', epochs=50, batch=16)
