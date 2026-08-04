import cv2

#read the image 
image = cv2.imread('messi.jpeg')

# detact the edges in the image using Canny edge detection
image_edges = cv2.Canny(image,100,200)

# show the image
cv2.imshow('messi', image)
cv2.imshow('messi_edges', image_edges)

# print image info
print(f"ndim of original image {image.ndim}")
print(image)
print(f"ndim of edge image {image_edges.ndim}")
print(image_edges)
# wait for a key press and close the image window
cv2.waitKey(0)

# destroy all windows
cv2.destroyAllWindows() 