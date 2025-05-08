import cv2
import numpy as np
from anpr import ANPR
from parking_counter import ParkingCounter
from config import *

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Initialize ANPR
    anpr = ANPR()
    
    # Define parking spaces (example coordinates - adjust based on your camera view)
    parking_spaces = [
        (100, 100, 200, 100),  # (x, y, width, height)
        (350, 100, 200, 100),
        (100, 250, 200, 100),
        (350, 250, 200, 100),
    ]
    
    # Initialize parking counter
    parking_counter = ParkingCounter(parking_spaces)
    
    # Wait for camera to initialize
    print("Initializing camera...")
    for _ in range(30):  # Skip first 30 frames
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return
    
    # Set initial background
    ret, frame = cap.read()
    if ret:
        parking_counter.set_background(frame)
    
    print("Starting main loop...")
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for display
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Process ANPR
        plate_regions, plate_texts = anpr.process_frame(frame)
        
        # Update background and detect occupied spaces
        parking_counter.update_background(frame)
        occupied_spaces = parking_counter.detect_occupied_spaces(frame)
        
        # Draw results
        display_frame = parking_counter.draw_parking_spaces(display_frame, occupied_spaces)
        
        # Draw detected plates
        for (x, y, w, h), text in zip(plate_regions, plate_texts):
            # Scale coordinates to display size
            x = int(x * DISPLAY_WIDTH / FRAME_WIDTH)
            y = int(y * DISPLAY_HEIGHT / FRAME_HEIGHT)
            w = int(w * DISPLAY_WIDTH / FRAME_WIDTH)
            h = int(h * DISPLAY_HEIGHT / FRAME_HEIGHT)
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
            cv2.putText(display_frame, text, (x, y - 10), FONT, FONT_SCALE, COLOR_GREEN, FONT_THICKNESS)
        
        # Display frame
        cv2.imshow('ANPR and Parking Counter', display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 