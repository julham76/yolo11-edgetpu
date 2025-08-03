from ultralytics import YOLO
import pandas as pd
import cv2
import glob
import os

model = YOLO('/home/pi/yolo-edgetpu/my_model_falldown_full_integer_quant_edgetpu.tflite', task='detect')
#model = YOLO('/home/pi/yolo-edgetpu/yolo11s.pt', task='detect')
#model = YOLO('5my_edgetpu.tflite', task='detect')

#results = model('8.png', verbose=False, conf=0.5)
#results = model('/home/pi/yolo-edgetpu/test1.jpg', verbose=False, conf=0.4)
image_folder = '/home/pi/yolo-edgetpu/jpg'
output_folder = '/home/pi/yolo-edgetpu/result_jpg'

image_files = glob.glob(os.path.join(image_folder, '*.jpg')) + \
              glob.glob(os.path.join(image_folder, '*.png')) + \
              glob.glob(os.path.join(image_folder, '*.jpeg'))
              
# Extract and store detection data
# all_detections = []

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load a pre-trained YOLOv8 model
#model = YOLO('yolov8n.pt')  # You can choose other models like 'yolov8s.pt', etc.

# Loop through each image and perform object detection
for image_path in image_files:
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        continue

    # Perform inference
    results = model(img)

    # Get the annotated image (with bounding boxes and labels)
    annotated_img = results[0].plot()

    # Save the annotated image to the output folder
    output_filename = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_filename, annotated_img)

    print(f"Processed and saved: {output_filename}")

print("Object detection complete for all images in the folder.")
