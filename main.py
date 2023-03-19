import cv2
import numpy as np
import time
import sys
import os


CONFIDENCE = 0.5

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("cfg/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("cfg/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    if cap.isOpened():
        ret, img = cap.read()
    cv2.imshow('before train', img)

    img = cv2.imread("src_room.jpg", cv2.IMREAD_UNCHANGED)

    height, width, channels = img.shape

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    img.shape: (1200, 1800, 3)
    blob.shape: (1, 3, 416, 416)

    net.setInput(blob)
    outs = net.forward(output_layers)

    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[:4]
            class_id = np.argmax(scores)
            confidence = int(scores[class_id])
            if confidence > CONFIDENCE:
                box = scores * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    for i in range(len(boxes)):
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = img.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    cv2.imshow('after train', img)

    if cv2.waitKey(1) == ord('q'):
        break