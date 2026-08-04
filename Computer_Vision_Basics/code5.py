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

    # convert the captured frame to grayscale
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detect the edges in the captured frame using Canny edge detection
    frame_edges = cv2.Canny(frame,100,200)

    #show the captured frame
    cv2.imshow('camera',frame)
    cv2.imshow('camera_gray',frame_gray)
    cv2.imshow('camera_edges',frame_edges)
    #wait till uses a key q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# wait for a key press and close the video window
cv2.waitKey(0)

# destroy all windows
cv2.destroyAllWindows()

# release the video capture object
cap.release()