from ultralytics import YOLO
import pandas as pd

model = YOLO('/home/pi/yolo-edgetpu/my_model_falldown_full_integer_quant_edgetpu.tflite', task='detect')
#model = YOLO('/home/pi/yolo-edgetpu/yolo11s.pt', task='detect')
#model = YOLO('5my_edgetpu.tflite', task='detect')

#results = model('8.png', verbose=False, conf=0.5)
results = model('/home/pi/yolo-edgetpu/test1.jpg', verbose=False, conf=0.4)

# Extract and store detection data
all_detections = []
for r in results:
    for box in r.boxes:
        # Extract bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Extract class ID and name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        # Extract confidence score
        confidence = float(box.conf[0])
        
        all_detections.append({
            'image_path': r.path,
            'class_id': class_id,
            'class_name': class_name,
            'confidence' : round(confidence,3),
            'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)
        })
    r.save(filename='/home/pi/yolo-edgetpu/detection.jpg')

# Save to a CSV file
df = pd.DataFrame(all_detections)
df.to_csv('/home/pi/yolo-edgetpu/detections.csv', index=False) 
df.to_json('/home/pi/yolo-edgetpu/detections.json', index=False)

#if r: r.release()

