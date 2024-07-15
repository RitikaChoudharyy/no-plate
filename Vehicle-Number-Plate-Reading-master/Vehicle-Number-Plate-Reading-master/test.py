import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time

# Load the image
image = cv2.imread('car.jpeg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or unable to load.")
    sys.exit()

# Resize the image
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to the grayscale image
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Detect edges in the image
edged = cv2.Canny(gray, 170, 200)

# Find contours in the edged image
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

NumberPlateCnt = None

# Loop over the contours to find the number plate contour
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  
        NumberPlateCnt = approx 
        break

# Check if the number plate contour is found
if NumberPlateCnt is None:
    print("Error: No contour found for the number plate.")
    sys.exit()

# Masking the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
cv2.imshow("Final_image", new_image)

# Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)

# Data is stored in CSV file
raw_data = {'date': [time.asctime(time.localtime(time.time()))], 
            'v_number': [text]}

df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
df.to_csv('data.csv', index=False)

# Print recognized text
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
