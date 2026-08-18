import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt 
import streamlit as st
from PIL import Image
from torchvision import transforms

# select the deveice
device = ""
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
print(f"Device = {device}")

model = nn.Sequential(
    # add the convolution2d layer
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),

    # configure ReLu activation function
    nn.ReLU(),

    # add max pooling layer
    nn.MaxPool2d(kernel_size=2, stride=2),

    # add one more convolution layer
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),

    # configure ReLu activation function
    nn.ReLU(),

    # add max pooling layer
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    # add the flatten layer
    nn.Flatten(),

    # add the hidden layer with 128 neurons
    nn.Linear(in_features=32*7*7, out_features=128),

    # add ReLu on the hidden layer
    nn.ReLU(),

    # add the output layer
    # since there are 10 digits the output layer will have 10 neurons
    nn.Linear(in_features=128, out_features=10)
)



state_dict = torch.load(
    "./mnist_model.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)
model = model.to(device)

st.set_page_config("Hand Written digit Recognation")

st.header("Hand Written digit Recognation")

st.write("Upload a image containing handwritten digit ")

file= st.file_uploader(label="Upload file",type=['png','jpg','jpeg'])

if file is not None:
    image = Image.open(file).convert("L")

    # convert pillow image to 3d array
    image = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image=image,clamp=True)

    #binarize the image
    return_value,binary_image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)

    points = cv2.findNonZero(binary_image)

    if points is None:
        st.error("No digit found, Please Upload the Right image")
    else:
        # Find the Bounding Image
        x,y,w,h = cv2.boundingRect(points)

        digit = binary_image[y:y+h,x:x+w]

        size = max(w,h)

        # Make the Square of the Size
        square = np.zeros((size,size),dtype=np.uint8)

        # Find the offset :
        x_offset = (size-w)//2
        y_offset = (size-h)//2

        square[y_offset:y_offset+h,x_offset:x_offset+w] = digit

        # Resize the image for the Models Requirement
        image_resized = cv2.resize(square,(28,28))

        # Create the Transformer to create the image to craete the tensor
        transformer = transforms.ToTensor()

        # convert The Image to The Tensor
        image = transformer(image_resized)

        # unsqueeze the image
        image = image.unsqueeze(0)

        # Move to the Device 
        image = image.to(device=device)

        with torch.no_grad():
            output=model(image)
            predicted_class = torch.argmax(output,dim=1).item()

            #Show to the User
            st.success(f"Predicted the Digit = {predicted_class}")


