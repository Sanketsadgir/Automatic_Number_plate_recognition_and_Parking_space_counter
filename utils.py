import cv2
import numpy as np
import pytesseract
from config import *

def preprocess_image(image):
    """Preprocess image for better plate detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    return thresh

def find_contours(image):
    """Find contours in the image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_plate_contours(contours):
    """Filter contours to find potential license plates."""
    plate_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Filter out very small contours
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # License plates typically have an aspect ratio between 2.0 and 5.0
        if 2.0 <= aspect_ratio <= 5.0:
            plate_contours.append((x, y, w, h))
            
    return plate_contours

def extract_plate_text(image, plate_region):
    """Extract text from the license plate region."""
    x, y, w, h = plate_region
    plate_image = image[y:y+h, x:x+w]
    
    # Preprocess the plate image
    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    
    # Apply thresholding
    _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract text using Tesseract
    text = pytesseract.image_to_string(plate_thresh, config=TESSERACT_CONFIG)
    return text.strip()

def draw_results(image, plate_regions, plate_texts, parking_spaces):
    """Draw detection results on the image."""
    # Draw plate regions and text
    for (x, y, w, h), text in zip(plate_regions, plate_texts):
        cv2.rectangle(image, (x, y), (x + w, y + h), COLOR_GREEN, 2)
        cv2.putText(image, text, (x, y - 10), FONT, FONT_SCALE, COLOR_GREEN, FONT_THICKNESS)
    
    # Draw parking spaces
    for space in parking_spaces:
        x, y, w, h, occupied = space
        color = COLOR_RED if occupied else COLOR_GREEN
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    return image

def calculate_parking_occupancy(frame, background):
    """Calculate parking space occupancy."""
    # Convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray, bg_gray)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh 