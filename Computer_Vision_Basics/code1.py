# import the required packages
import cv2
import numpy as np

# read the image
image = cv2.imread('messi.jpeg')

# get image details
print(type(image))
print(f"no of dimensions {image.ndim}")
print(f"shape of image {image.shape}")
print(f"size of image {image.size}")
print(f"data type of image {image.dtype}")
print(f"memory size of image {image.nbytes/(1024*1024)} MB")


# OpenCV uses BGR color space by default, so we need to convert it to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

(height, width, channels) = image.shape
print(f"height * width * channels = {height} * {width} * {channels}")