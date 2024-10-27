from ultralytics import YOLO
model = YOLO('yolov8s-world')
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16, patience=50)