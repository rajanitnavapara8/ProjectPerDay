import cv2
import mediapipe as mp
import time

# mediapipe.python.solutions.objectron.BoxLandmark
class handDetector():

    def __init__(self, mode=False, maxHands=2, DetenctionCon=0.5,TrackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.DetenctionCon = DetenctionCon
        self.TrackingCon = TrackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.DetenctionCon, self.TrackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingers = [8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0,draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                self.lmList.append([id,cx,cy])
                # if id==12:
                if draw:
                    # cv2.circle(img, (cx,cy), 10, (205,150,155),cv2.FILLED)
                    cv2.circle(img, (cx,cy), 5, (0,10,155),cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        finglist = []
        # img = cv2.flip(img, 1)
        # print(overlay[0].shape)
        # w,h,c = overlay[0].shape
        # img[0:w,0:h]=overlay[0]
        # img = detector.findHands(img)
        # self.lmList = detector.findPosition(img, draw=False)
        # print(lmList)

        if len(self.lmList) != 0:
            # print(lmList[8])
            if self.lmList[4][1] > self.lmList[17][1] and self.lmList[4][1] > self.lmList[3][1]:
                finglist.append(1)
            else:
                finglist.append(0)
            if self.lmList[4][1] < self.lmList[17][1] and self.lmList[4][1] < self.lmList[3][1]:
                finglist.append(1)
            else:
                finglist.append(0)
            for i in self.fingers:
                if self.lmList[i][2] < self.lmList[i - 1][2]:
                    # print(f"{i} is up.")
                    finglist.append(1)
                else:
                    finglist.append(0)

        return finglist[1:]

def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            # pads
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3 ,(255,0,255),3)

        cv2.imshow("Image",img)
        
        if cv2.waitKey(1) & 0xff==ord('d'):
            break

    # vdo.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
        