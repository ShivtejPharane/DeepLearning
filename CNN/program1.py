import numpy as np
import cv2

# Load the image
image = cv2.imread('cat.jpeg')

def convert_to_grayscale(image):
    # Convert the image to grayscale
    greayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale :",greayimage )

def resize_the_image(image):
    image_resized = cv2.resize(image,(100,100))
    cv2.imshow("Resized image",image_resized)

def sharpen_image(image):
    # Sharpening means enhance the edges
    # Sharpen the image kernal matrix
    kernal =np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,1,0]
    ])
    image_sharpen = cv2.filter2D(image,-5,kernel=kernal)
    cv2.imshow("Sharpen ",image_sharpen)

def emboss_image(image):
    # Kernal
    kernal = np.array([
        [-2,-1,0],
        [-1,1,1],
        [0,1,2]
    ])
    embossed_image = cv2.filter2D(image,-1,kernal)
    cv2.imshow("Embossed image",embossed_image)

def blur_image(image):
    kernal = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])/9
    blur_image = cv2.filter2D(image,-1,kernal)
    cv2.imshow("Blured",blur_image)

def edge_detection(image):
    kernal = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ])
    edge_detected = cv2.filter2D(image,-1,kernal)
    cv2.imshow("Edge Detected",edge_detected)
#Show the original image
cv2.imshow('Original Image', image)
convert_to_grayscale(image)
resize_the_image(image)
sharpen_image(image)
emboss_image(image)
blur_image(image=image)
edge_detection(image=image)
cv2.waitKey(0)
cv2.destroyAllWindows()
