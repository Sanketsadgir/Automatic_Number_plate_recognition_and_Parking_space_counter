import cv2
import numpy as np
from utils import preprocess_image, find_contours, filter_plate_contours, extract_plate_text
from config import *

class ANPR:
    def __init__(self):
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    
    def detect_plates(self, frame):
        """Detect license plates in the frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect plates using Haar cascade
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_PLATE_WIDTH, MIN_PLATE_HEIGHT),
            maxSize=(MAX_PLATE_WIDTH, MAX_PLATE_HEIGHT)
        )
        
        plate_regions = []
        plate_texts = []
        
        for (x, y, w, h) in plates:
            # Extract and process the plate region
            plate_region = (x, y, w, h)
            plate_text = extract_plate_text(frame, plate_region)
            
            # Only add if text was successfully extracted
            if plate_text:
                plate_regions.append(plate_region)
                plate_texts.append(plate_text)
        
        return plate_regions, plate_texts
    
    def process_frame(self, frame):
        """Process a single frame for license plate detection and recognition."""
        # Preprocess the image
        processed = preprocess_image(frame)
        
        # Find and filter contours
        contours = find_contours(processed)
        plate_contours = filter_plate_contours(contours)
        
        # Extract text from potential plates
        plate_regions = []
        plate_texts = []
        
        for contour in plate_contours:
            text = extract_plate_text(frame, contour)
            if text:
                plate_regions.append(contour)
                plate_texts.append(text)
        
        return plate_regions, plate_texts 