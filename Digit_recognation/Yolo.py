from ultralytics import YOLO
import cv2

# Create the Yolo Object and downdload the required pre-trained model
model = YOLO("yolo26n.pt")

# Load the Test image
test_image = "./test.webp"

image = cv2.imread(test_image)

results = model(test_image)

# get the first result
result = results[0]

# Find the bounding rectangles of the detected objects
for box in result.boxes:
    # get the box id
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    #print(f"class id : {class_id}, Class name : {class_name}")
    
    # Get the confidence of the detected object
    confidence = float(box.conf[0])
    print(f"class id : {class_id}, Class name : {class_name}, Confidence : {confidence:.2f}")

# render the detected objects from the image
annotated_image = result.plot()

# Show the renderd image to the user
cv2.imshow("objects detected",annotated_image)
cv2.imshow("Orignal Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()