import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt 
import streamlit as st



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
