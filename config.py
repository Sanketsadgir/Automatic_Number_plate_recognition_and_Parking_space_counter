# Camera settings
CAMERA_ID = 0  # Default camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ANPR settings
MIN_PLATE_WIDTH = 60
MIN_PLATE_HEIGHT = 20
MAX_PLATE_WIDTH = 200
MAX_PLATE_HEIGHT = 100

# Parking space detection settings
PARKING_SPACE_THRESHOLD = 0.3  # Threshold for considering a space occupied
MOTION_THRESHOLD = 25  # Threshold for motion detection
MIN_CONTOUR_AREA = 1000  # Minimum contour area to consider as a vehicle

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FONT_SCALE = 0.7
FONT_THICKNESS = 2
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Tesseract settings
TESSERACT_CONFIG = '--psm 7 --oem 3'  # Page segmentation mode 7 (treat as single line) 