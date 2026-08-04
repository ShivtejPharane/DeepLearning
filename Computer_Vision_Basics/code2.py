import cv2

#read the image 
image = cv2.imread('messi.jpeg')

# show the image
cv2.imshow('messi', image)

# wait for a key press and close the image window
cv2.waitKey(0)

# destroy all windows
cv2.destroyAllWindows() 