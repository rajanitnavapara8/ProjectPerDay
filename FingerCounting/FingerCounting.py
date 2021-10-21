import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import time
import os

foldpath = "handImg"
l1 = os.listdir(foldpath)
print(l1)
overlay = []
for i in l1:
    img = cv2.imread(f'{foldpath}/{i}')
    overlay.append(img)

print(len(overlay))
################################
wCam, hCam = 1280,720
#####s###########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)    
cap.set(4, hCam)
pTime = 0
 
detector = htm.handDetector(DetenctionCon=0.8)
while True:
    success,img = cap.read()
    img = cv2.flip(img,1)   
    # print(overlay[0].shape)
    # w,h,c = overlay[0].shape  
    # img[0:w,0:h]=overlay[0]
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    # print(lmlist)
    fingers = [8,12,16,20]
    if len(lmlist) != 0:
        # print(lmlist[8])
        finglist = []
        if lmlist[4][1] > lmlist[17][1] and lmlist[4][1] > lmlist[3][1]:
            finglist.append(1)
        else:
            finglist.append(0)
        if lmlist[4][1] < lmlist[17][1] and lmlist[4][1] < lmlist[3][1]:
            finglist.append(1)
        else:   
            finglist.append(0)
        for i in fingers:
            if lmlist[i][2] < lmlist[i-1][2]:
                # print(f"{i} is up.")
                finglist.append(1)
            else:
                finglist.append(0)
        # print(finglist)
        count = finglist.count(1)
        w,h,c = overlay[count-1].shape  
        img[0:w,0:h]=overlay[count-1]
        cv2.rectangle(img,(50,250),(200,450),(255,0,0),cv2.FILLED)
        cv2.putText(img,f'{count}',(75,400),cv2.FONT_HERSHEY_PLAIN,10,(0,255,0),10)
        cv2.putText(img,f'{finglist[1:]}',(50,430),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1000, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
 
    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xff==ord('d'):
        break

# vdo.release()
cv2.destroyAllWindows()