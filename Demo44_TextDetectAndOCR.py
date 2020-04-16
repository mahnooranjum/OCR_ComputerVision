#==============================================================================
#   By: Mahnoor Anjum
#   Date: 16/04/2020
#   Codes inspired by:
#   Official Documentation
#   GeeksforGeeks.com
#==============================================================================
def show(image):
    cv2.imshow("Show", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()                         

# import the necessary packages
# Import required packages 
import cv2 
import pytesseract 

# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe" 

# Read image from which text needs to be extracted 
img = cv2.imread("imgs/demo42.jpg")

# Preprocessing the image starts 

# Convert the image to gray scale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# Performing OTSU threshold 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 

# Specify structure shape and kernel size. 
# Kernel size increases or decreases the area 
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect 
# each word instead of a sentence. 
n = 5
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n)) 

# Applying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
show(dilation)

# Finding contours 
_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_NONE) 

# Creating a copy of image 
im2 = img.copy() 


# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
for cnt in contours: 
	x, y, w, h = cv2.boundingRect(cnt) 
	
	# Drawing a rectangle on copied image 
	rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
	
	# Cropping the text block for giving input to OCR 
	cropped = im2[y:y + h, x:x + w] 
	
	
	# Apply OCR on the cropped image 
	text = pytesseract.image_to_string(cropped) 
	print(text)
