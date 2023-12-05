import cv2
import numpy as np
import time


# Load the pre-trained Haar cascade for vehicles
vehicle_cascade = cv2.CascadeClassifier('cars.xml')


# Define the Traffic Signal Color
RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (0,255,255)

#Initialized the Adjustable alloted Time for Traffic Signal

allotted_time = 20

# Start the Timer

start_time = time.time()

# Initializing Traffic Signal State

traffic_signal_state = RED


def detect_vehicles(frame_gray):
    vehicles = vehicle_cascade.detectMultiScale(frame_gray, 1.1, 5)
    return vehicles



def detect_traffic_congestion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles in the frame
    vehicles = detect_vehicles(gray)


    # Draw rectangles around the detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

   

    # Check the number of vehicles and traffic lights detected
    num_vehicles = len(vehicles)
  

    # Check if there is traffic congestion
    if num_vehicles > 10:
        traffic_signal_state = GREEN
        cv2.putText(frame, "Traffic Congestion Detected", (100,250), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    elif num_vehicles < 10:
        traffic_signal_state = RED
        cv2.putText(frame, "No Traffic Congestion Detected", (100,250), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
    else:
        traffic_signal_state = GREEN
        

    # Draw traffic signal frame
    cv2.circle(frame, (100,100), 50, traffic_signal_state,-1)
    
   

# Capture video from the webcam or video file
cap = cv2.VideoCapture('6.mp4')


while True:
    
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        break

    # Detect traffic congestion in the frame
    congestion_frame = detect_traffic_congestion(frame)

    # Display the frame with traffic congestion detection
    cv2.imshow('Computerized Traffic Management System', congestion_frame)
    
    
        
        

    # Wait for 'q' key to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()