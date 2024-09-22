import cv2

capture = cv2.VideoCapture("C:/Users/Adhay/OneDrive/Desktop/Open CV/Lesson 10 homework/cars.mp4")

car_cascade = cv2.CascadeClassifier("C:/Users/Adhay/OneDrive/Desktop/Open CV/Lesson 10 homework/carplate.xml")


while True:

    ret,frames = capture.read()
    grayscale = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    detect = car_cascade.detectMultiScale(grayscale, 1.1,1)
    #face_cascade.detectMultiScale(image, scaleFactor, Min Neighbours)
    for (x,y,w,h) in detect:

        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Car Detection", frames)
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()