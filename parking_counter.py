import cv2
import numpy as np
from utils import calculate_parking_occupancy
from config import *

class ParkingCounter:
    def __init__(self, parking_spaces):
        """
        Initialize the parking counter.
        parking_spaces: List of tuples (x, y, w, h) defining parking space regions
        """
        self.parking_spaces = parking_spaces
        self.background = None
        self.occupied_spaces = set()
    
    def set_background(self, frame):
        """Set the background frame for motion detection."""
        self.background = frame.copy()
    
    def update_background(self, frame, alpha=0.1):
        """Update the background frame using running average."""
        if self.background is None:
            self.background = frame.copy()
        else:
            cv2.accumulateWeighted(frame, self.background, alpha)
    
    def detect_occupied_spaces(self, frame):
        """Detect which parking spaces are occupied."""
        if self.background is None:
            return []
        
        # Calculate motion mask
        motion_mask = calculate_parking_occupancy(frame, self.background)
        
        # Check each parking space
        occupied_spaces = []
        for i, (x, y, w, h) in enumerate(self.parking_spaces):
            # Extract the region of interest
            roi = motion_mask[y:y+h, x:x+w]
            
            # Calculate the percentage of motion in the space
            motion_percentage = np.sum(roi > 0) / (w * h)
            
            # Determine if the space is occupied
            is_occupied = motion_percentage > PARKING_SPACE_THRESHOLD
            occupied_spaces.append((x, y, w, h, is_occupied))
            
            if is_occupied:
                self.occupied_spaces.add(i)
            else:
                self.occupied_spaces.discard(i)
        
        return occupied_spaces
    
    def get_available_spaces(self):
        """Get the number of available parking spaces."""
        return len(self.parking_spaces) - len(self.occupied_spaces)
    
    def draw_parking_spaces(self, frame, occupied_spaces):
        """Draw parking spaces on the frame."""
        for space in occupied_spaces:
            x, y, w, h, is_occupied = space
            color = COLOR_RED if is_occupied else COLOR_GREEN
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Add text showing space status
            status = "Occupied" if is_occupied else "Available"
            cv2.putText(frame, status, (x, y - 10), FONT, FONT_SCALE, color, FONT_THICKNESS)
        
        # Add total count
        total_spaces = len(self.parking_spaces)
        available_spaces = self.get_available_spaces()
        cv2.putText(frame, f"Available: {available_spaces}/{total_spaces}", 
                   (10, 30), FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
        
        return frame 