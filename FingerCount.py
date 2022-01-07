import cv2
import time
import mediapipe as mp
import HandTrackingModule1 as htm
import numpy as np

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    while True:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            print(lmlist)
            print(lmlist[4])
            if lmlist[4][2]>lmlist[3][2]:
                cv2.putText(img,"Thumbs Down",(30,100),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            else:
                cv2.putText(img,"Thumbs Up",(30,100),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        # if keyPressed == ord('q'):
        #     break;
if __name__=="__main__":
    main()