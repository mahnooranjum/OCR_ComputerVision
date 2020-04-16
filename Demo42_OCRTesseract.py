#==============================================================================
#   By: Mahnoor Anjum
#   Date: 16/04/2020
#   Codes inspired by:
#   Official Documentation
#   PyImageSearch.com
#==============================================================================

import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe" 

img = cv2.imread("imgs/demo42.jpg")
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
test_image = Image.fromarray(img)
text = tess.image_to_string(test_image, lang='eng')
print("PyTesseract Detected the following text: ", text)