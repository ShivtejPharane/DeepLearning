import cv2

# capture the video from the source 
# source :
# - number (index of the camera)
# - url (path to the video file)
cap = cv2.VideoCapture(0)

# read the video frame by frame
while True:
    # read the captured frame
    ret,frame = cap.read()

    #show the captured frame
    cv2.imshow('camera',frame)

    #wait till uses a key q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# wait for a key press and close the video window
cv2.waitKey(0)

# destroy all windows
cv2.destroyAllWindows()

# release the video capture object
cap.release()