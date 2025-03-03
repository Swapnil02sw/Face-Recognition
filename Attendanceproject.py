import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path='ImageStudent'
images=[]
studentNames=[]
mylist=os.listdir(path)
print(mylist)
for Student in mylist:
    curImg=cv2.imread(f'{path}/{Student}')
    images.append(curImg)
    studentNames.append(os.path.splitext(Student)[0])
print(studentNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        studentDataList=f.readlines()
        studentNamesList=[]
        for line in studentDataList:
            entry=line.split(',')
            studentNamesList.append(entry[0])
        if name not in studentNamesList:
            now= datetime.now().strftime('%H:%M:%S')
            dt=datetime.today().strftime('%d:%m:%y')
            f.writelines(f'\n{name},{dt},{now}')

encodeListKnown=findEncodings(images)
camp=cv2.VideoCapture(0)

while True:
    Success,img=camp.read()
    Simg=cv2.resize(img,(0,0),None,0.25,0.25)
    Simg=cv2.cvtColor(Simg,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(Simg)
    encodesCurFrame=face_recognition.face_encodings(Simg,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=studentNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-40),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+2,y2-2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
