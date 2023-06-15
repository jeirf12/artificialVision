import cv2
import numpy as np
from ultralytics import YOLO
import pandas
import matplotlib.path as mplPath

ZONE = np.array([
    [7, 12],
    [623, 14],
    [621, 462],
    [15, 456],
])


def detection(cap):
    while cap.isOpened():
        status, frame = cap.read()
        if not status: break
        model = load_model()
        detect = model.predict(frame, imgsz=640, conf=0.6)
        classkey, bboxes, confidences = get_data(detect)
        print('Classes: ', classkey)
        print('Boxes: ', bboxes)
        print('Confidence: ', confidences)
        detectionsLed = 0
        detectionsCap = 0
        colorBox = (0, 0, 255)
        nameClass = ""
        for ind, box in enumerate(bboxes):
            xc, yc = get_center(box)
            if is_valid(xc, yc):
                if classkey[ind] == 0:
                    detectionsLed += 1
                    nameClass = "Led"
                elif classkey[ind] == 1:
                    detectionsCap += 1
                    nameClass = "Capacitor"
                    colorBox = (255, 0, 0)
            # cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0,255,0), thickness=-1)
            cv2.putText(img = frame, text = f"{nameClass}: {confidences[ind]}", org=(xc, yc - 10), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 3, color = colorBox, thickness = 1)
            cv2.rectangle(img = frame, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(255, 0, 0), thickness=1)
        cv2.putText(img = frame, text = f"Led: {detectionsLed}", org=(100, 40), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 3, color = (0,0,255), thickness = 4)
        cv2.putText(img = frame, text = f"Capacitor: {detectionsCap}", org=(100, 100), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 3, color = (255,0,0), thickness = 4)
        # cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(0,255,0), thickness=4)

        cv2.imshow("frame", frame)
        if (cv2.waitKey(10) & 0xFF) == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()


def load_model():
    # En el name_model.pt se carga el modelo que desea entrenar o hacerle transfer learn   
    model = YOLO('name_model.pt')
    model.fuse()
    return model


def get_data(detect):
    classkey = detect[0].boxes.cls.cpu().numpy()
    bboxes = detect[0].boxes.xyxy.cpu().numpy().astype(int)
    confidences = detect[0].boxes.conf.cpu().numpy()
    return classkey, bboxes, confidences


def get_center(bbox):
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center


def is_valid(xc, yc):
    return mplPath.Path(ZONE).contains_point((xc, yc))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detection(cap)


