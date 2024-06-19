# Smart_parking_system
This is the implementation of Vehicle detection and Number plate detection.
#main.py
main.py uses yolo v8 to detect vehicles and capture the dimensions of the vehicle. Each vehicle is given an id and all four points of the vehicle box. This data is then used by ocr to detect the number plate for the vehicle. If a number plate is detected, it 
#util.py 
util.py contains all the required functions for to implement it. The functions are read_license_plate, which detects the number plates using the bounding box obtained by yolo. The detected number plate will be stored as text. The function license_complies_format checks if the text is in the proper format or not. If the text is in the proper format, then
