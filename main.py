import mysql.connector
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

# Initialize variables
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('C:/Users/Daniel/Desktop/Coding/VRITIKA_Internship/number_plate_detection/models/license_plate_detector.pt')

# Connect to MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Dany@0579",
    database="xam"
)

# Create cursor
mycursor = mydb.cursor()

# Load video
cap = cv2.VideoCapture('./parklowcut.mp4')
vehicles = [2, 3, 5, 7]

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(detections_)

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

                    # Insert data into database
                    sql = "INSERT INTO chair (frame_nmr, car_id, car_bbox_x1, car_bbox_y1, car_bbox_x2, car_bbox_y2, " \
                          "license_plate_bbox_x1, license_plate_bbox_y1, license_plate_bbox_x2, license_plate_bbox_y2, " \
                          "license_plate_bbox_score, license_number, license_number_score) " \
                          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

                    val = (frame_nmr, car_id,
                           results[frame_nmr][car_id]['car']['bbox'][0], results[frame_nmr][car_id]['car']['bbox'][1],
                           results[frame_nmr][car_id]['car']['bbox'][2], results[frame_nmr][car_id]['car']['bbox'][3],
                           results[frame_nmr][car_id]['license_plate']['bbox'][0], results[frame_nmr][car_id]['license_plate']['bbox'][1],
                           results[frame_nmr][car_id]['license_plate']['bbox'][2], results[frame_nmr][car_id]['license_plate']['bbox'][3],
                           results[frame_nmr][car_id]['license_plate']['bbox_score'], results[frame_nmr][car_id]['license_plate']['text'],
                           results[frame_nmr][car_id]['license_plate']['text_score'])

                    mycursor.execute(sql, val)

# Commit changes to the database
mydb.commit()

# Close cursor and connection
mycursor.close()
mydb.close()
