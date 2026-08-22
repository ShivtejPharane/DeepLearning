from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image


model = YOLO("yolo26n.pt")

st.set_page_config(page_title="Object Detection")

st.header("Object Detection")
st.subheader("Upload an Image for the object Detection")

file = st.file_uploader(label="Upload the File",type=['png','jpg','jpeg','webp'])

if file is None:
    st.error("Please Upload the file for the Object Detection")
else:
    image = Image.open(file).convert("RGB")

    # Divide the Image into 2 columns 
    col1,col2 = st.columns(2)

    #Col1 contents
    with col1:
        st.image(image=image,caption="Orignal Image",clamp=True)

    with col2:
        with st.spinner("detecting the objects..."):
            results = model(image)
            result = results[0]

            #Get the annotated image 
            annotated_image = result.plot(pil=True)

            st.image(annotated_image,caption="Detected Images",clamp=True)

