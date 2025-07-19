import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time


model = YOLO('yolo11n_full_integer_quant_edgetpu.tflite')


cap = cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

#frame_count = 0
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1020,600))
    if not ret:
        break
        
    #frame_count += 1
    #if frame_count % 3 != 0:
    #    continue

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
     
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)        
    cv2.imshow("FRAME", frame)
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
