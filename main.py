
import cv2 as cv
import math
import time
import argparse
import numpy as np

def getFaceBox(net,frame,conf_threshold=0.5):
    frameOpencvDnn=frame.copy() 
    (h,w)=frameOpencvDnn.shape[:2]

    blob=cv.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),(104.0,177.0,123.0),True,False)
    net.setInput(blob)
    detections=net.forward()
    bboxes=[]

    #loop over the detections
    for i in range (0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > conf_threshold:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h]) 
            (x1,y1,x2,y2)=box.astype("int")
            bboxes.append([x1,y1,x2,y2])
            # rectangle(img, pt1, pt2, color, thickness=None, lineType=None)
            cv.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(h/150)),8)

    return frameOpencvDnn,bboxes # returns frame and list of no of coordinates of each box

parser=argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')

parser.add_argument("-i",help="Path to input image or video file. Skip this argument to capture frames from a camera.")

args=parser.parse_args() #object

# CNN consists of 8 values for 8 age classes (“0–2”, “4–6”, “8–13”, “15–20”, “25–32”, “38–43”, “48–53” and “60-100”)
# here our model is pretrained in such a way that it can give a range of values weather this person age is between x to y (prediction of age)
agelist=['(0-2)' , '(4-6)' , '(8-12)' , '(15-20)' , '(25-32)' , '(38-43)' , '(48-53)' , '(60-100)']
genderlist=['Male','Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageNet=cv.dnn.readNetFromCaffe(ageProto,ageModel)
genderNet=cv.dnn.readNetFromCaffe(genderProto,genderModel)

faceNet=cv.dnn.readNet(faceModel,faceProto)


cap=cv.VideoCapture(args.i if args.i else 0) #args.i is image if it is there then it will be captured
padding = 20 
while cv.waitKey(1) < 0: 
    t=time.time()
    hasFrame,frame=cap.read()
    
    if not hasFrame: 
        cv.waitKey() 
        break

    frameface,bboxes=getFaceBox(faceNet,frame)

    #if no faces are detected
    if not bboxes:
        print("No face detected , Checking next frame")
        continue

    for bbox in bboxes: 
        face=frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        blob=cv.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds=genderNet.forward() 

        gender=genderlist[genderPreds[0].argmax()]

        print("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=agelist[agePreds[0].argmax()]

        print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))
        label="{},{}".format(gender,age)
        cv.putText(frameface, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)

        cv.imshow("Age Gender Demo", frameface) 
        if args.i:
            name=args.i
            cv.imwrite('./detected/'+name,frameface)
    
    print("Time : {:.3f}".format(time.time() - t))