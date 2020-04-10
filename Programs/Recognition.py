def openCycle():
	print('Door will be open. Please wait for few seconds')
	time.sleep(5)
	p.ChangeDutyCycle(7.5)  # turn towards 90 degree
	time.sleep(2)
        print('Door is opened')
	time.sleep(10)
	p.ChangeDutyCycle(2.5)
	time.sleep(2)
	print('Door is closed')
	GPIO.cleanup()
	
# Import OpenCV2 for image processing
import cv2
import time
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BOARD)

GPIO.setup(12, GPIO.OUT)

p = GPIO.PWM(12, 50)

p.start(0)

time.sleep(2) 

print('Project started')

time.sleep(2)

p.ChangeDutyCycle(2.5)

time.sleep(2)

import numpy as np

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('/home/pi/Raspberry-Face-Recognition-master/trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "/home/pi/Raspberry-Face-Recognition-master/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.3,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        #cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('im',im)
        # Recognize the face belongs to which ID
        qwe = recognizer.predict(gray[y:y+h,x:x+w])
        print qwe
        ty=''
        if(qwe[1]<60):
        # Check the ID if exist 
            if(qwe[0] == 2):
                print "Hema"
		openCycle()
                ty="Hema"
            #If not exist, then it is Unknown
            elif(qwe[0] == 5):
                print "Varun"
                openCycle()
                ty="Varun"
	    elif(qwe[0] == 1):
                print "Sarma"
                openCycle()
                ty="Sarma"	
            else:
                print ('know')
		p.ChangeDutyCycle(2.5)  # turn towards 90 degree
		time.sleep(2)
                ty="unkown"
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(ty), (x,y-40), font, 2, (255,255,255), 3)
        # Put text describe who is in the picture
	# Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()
p.stop()
# Close all windows
cv2.destroyAllWindows()
