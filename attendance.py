import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Image Basics' #stored image directory
images = [] #New list of images
classNames = [] # New list of students' names
myList = os.listdir(path) #get the lists from the path given
print(myList)
for cls in myList: #looping through the images in the directory
    curImg = cv2.imread(f'{path}/{cls}') # read through every image in the directory
    images.append(curImg) #append the images into images list
    classNames.append(os.path.splitext(cls)[0]) #Only store the name of the image without the format .jpg

print(classNames)

#attendance function to mark the attendance based on face detection and matching
def markattendance(name):
    with open('Attendance.csv', 'r+') as f: #open up the attendance file,and read/write it
        myDataList = f.readlines() #read the lines in the csv file
        nameList = [] #create an empty namelist
        for line in myDataList: #loop through each line in the csv file
            entry = line.split(',') # separate data entry between the two columns
            nameList.append(entry[0]) #append the namelist on the first column (NAME column)
        if name not in nameList: #this is to write new names from the images. Existing name will not be written or overwritten due to these lines of codes!
            now = datetime.now() #we set the time when the image is matched
            dtString = now.strftime('%H:%M:%S') # set the time formatting
            f.writelines(f'\n{name},{dtString}') #write the name on the list, with the time




#encoding process
def findEncodings(images): # a function to find the encoding from a given list of images
    encodeList = [] #we create an empty list for the encodings
    for img in images: # we loop through the list of images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert the image to RGB
        encode = face_recognition.face_encodings(img)[0] #Gives encoding of the first face in the image
        encodeList.append(encode) # Append the encodings into the list
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete!') # To notify that the encoding function has been completed

cap = cv2.VideoCapture(0) # open up webcam

while True:
    success, img = cap.read() # This will read the video
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #scaling down the image to fasten the process
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #convert the video to rgb

    faceCurFrame = face_recognition.face_locations(imgS) #find faces in the video
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame) #get the encoding for the faces

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame): #loop through the face location and encoding
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # compare faces between stored face images and face images from camera
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # get the euclidean distance between the two images
        # print(faceDis)
        matchIndex = np.argmin(faceDis) #chooses the lowest face distance value as the best match, and give the index of that match (from the list of images)

        # labelling face after matching
        if matches[matchIndex]:
            name = classNames[matchIndex].upper() #set the name to be the name of the matched picture
            # print(name)
            y1,x2,y2,x1 = faceLoc #we set the coordinate to the face location
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #scale up the image to its original size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) #set the rectangle box to the face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED) #set a filled rectangle box for name
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #set the name for the matched picture
            markattendance(name) #mark the attendance

    cv2.imshow('Attendance Camera', img) #Show a window of webcam, together with the image matches
    if cv2.waitKey(1) & 0xFF == ord('q'): #this is run the video, and press 'q' on the webcam window to stop the program
        break
