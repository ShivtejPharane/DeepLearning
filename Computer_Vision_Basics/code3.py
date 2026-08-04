import cv2

image = cv2.imread('messi.jpeg')

# convert the RGB Image to Grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow('messi', image_gray)
cv2.imshow("orignal",image)

print(f"shape of image_orignal {image.shape}")
print(f"shape of gray image {image_gray.shape}")

print(f"dimensions of image_orignal {image.ndim}")
print(f"dimensions of gray image {image_gray.ndim}")

print(image[0])
print("-"*80)
print(image_gray[0])

#print no of pixels in the image
print(f"no of pixels in the image {image.size} = {image.shape[0]} * {image.shape[1]} * {image.shape[2]}")
print(f"no of pixels in the gray image {image_gray.size} = {image_gray.shape[0]} * {image_gray.shape[1]}")

cv2.waitKey(0)
cv2.destroyAllWindows()
