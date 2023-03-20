import cv2
import numpy as np
import time
import sys
import os

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("cfg/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("cfg/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
print(output_layers)

class_ids = []
confidences = []
boxes = []

def detect_img(frame):
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True,
                                 crop=False)
    net.setInput(blob)
    outs = net.forward()
    return outs

while True:
    _, img = cap.read()
    if cap.isOpened():
        height, width, _ = img.shape
        #outs = detect_img(img)
        cv2.imshow("Before train", img)

       for detection in outs:
                class_id = np.argmax(detection)
                confidence = detection[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + 30), color, -1)
                cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                cv2.imshow("After train", img)

cap.release()
cv2.destroyAllWindows()
