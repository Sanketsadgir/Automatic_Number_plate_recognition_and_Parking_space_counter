# ANPR and Parking Space Counter

This project implements an Automatic Number Plate Recognition (ANPR) system combined with a parking space counter using computer vision techniques.

## Features
- Real-time vehicle number plate detection and recognition
- Parking space availability tracking
- Software-only implementation using OpenCV
- No hardware dependencies required

## Requirements
- Python 3.8+
- OpenCV
- Tesseract OCR
- Other dependencies listed in requirements.txt

## Installation
1. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main script:
```bash
python main.py
```

2. Press 'q' to quit the application

## Project Structure
- `main.py`: Main application entry point
- `anpr.py`: Number plate recognition module
- `parking_counter.py`: Parking space detection module
- `utils.py`: Utility functions
- `config.py`: Configuration settings

## License
MIT License 