import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change to another number if you have multiple webcams

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

# Function to process frames from the webcam
def process_frame(frame):
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    
    # Display the resized frame (optional)
    cv2.imshow("Webcam Frame", frame)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to the grayscale image
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Detect edges in the image
    edged = cv2.Canny(gray, 170, 200)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    NumberPlateCnt = None
    
    # Loop over the contours to find the number plate contour
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break
    
    # Check if the number plate contour is found
    if NumberPlateCnt is not None:
        # Masking the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Configuration for tesseract
        config = '-l eng --oem 1 --psm 3'
        
        # Run tesseract OCR on image
        text = pytesseract.image_to_string(new_image, config=config)
        
        # Data to store in CSV file
        timestamp = time.asctime(time.localtime(time.time()))
        raw_data = {'date': [timestamp], 'v_number': [text]}
        
        # Create or append to CSV file
        df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
        df.to_csv('data.csv', mode='a', header=not os.path.exists('data.csv'), index=False)
        
        # Print recognized text
        print("Recognized Number Plate:", text)
    
    # Display the processed image with detected contours (optional)
    cv2.imshow("Processed Frame", new_image)

# Main loop to capture frames from the webcam
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break
    
    # Process the frame
    process_frame(frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
