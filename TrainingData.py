import cv2
import time
import mediapipe as mp
import HandTrackingModule1 as htm
import numpy as np
import csv
import pandas as pd

from math import dist, sqrt
import json 
import random

def getFpsColor(fps):
    if fps<=10:
        return (0,0,150)
    if fps>=30:
        return (0,150,0)
    g = min(150,(fps-10)*15)
    r = min(150,150-((fps-20)*15))
    return (0,g,r)
    

def getCenterOfMass(lmList):
    sumX = 0
    for i in range(21):
        sumX = sumX + lmList[i][1]
    sumY = 0
    for i in range(21):
        sumY = sumY + lmList[i][2]

    return sumX/21, sumY/21

def findDistance(x1,y1,x2,y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def getDistanceFromCenter(lmList):
    comX, comY = getCenterOfMass(lmList)
    distFromCOM = [0 for i in range(21)]

    for i in range(21):
        distFromCOM[i] = findDistance(comX, comY, lmList[i][1], lmList[i][2])

    return distFromCOM

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    countLabel = 0

    p = dict()
    targetLabel = "up"
    sampleSize = 500
    p['index']=[targetLabel+"_" + str(i) for i in range (sampleSize)]

    while countLabel<sampleSize:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            distfromCOM = getDistanceFromCenter(lmlist)

            for i in range(0,21):
                if i in p:
                    p[i].append(distfromCOM[i])
                else:
                    p[i]=[distfromCOM[i]]

            countLabel=countLabel+1
            
            #print(lmlist)
            #print(distfromCOM)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
        cv2.putText(img, "Frames taken: "+str(countLabel), (310,30), cv2.FONT_HERSHEY_PLAIN, 2, (150,0,0), 2)
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        # if keyPressed == ord('q'):
        #     break;

    # print(p)
    # print("\n")
    df = pd.DataFrame(p)
    df.insert(22,"Label", [targetLabel for i in range(sampleSize)])
    print(df)
    df.to_csv('trainingData\\'+targetLabel+'_trainingdata.csv')

if __name__=="__main__":
    main()