import cv2
from ultralytics import YOLOWorld

model = YOLOWorld('yolov8s-world')  

def detect_alcohol_in_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = box.conf
                label = result.names[class_id]
                
                if label in ['wine','beer']: 
                    x1, y1, x2, y2 = map(int, box.xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Alcohol Detection', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_alcohol_in_video()