import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detectConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:  # extract hand features from multiples hands
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)  # draw hand landmarks
        return img

    def findpositions(self, img, handno=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[handno]
            for ind, lm in enumerate(hands.landmark):
                # print(ind, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(ind, cx, cy)
                lmlist.append([ind, cx, cy])
                # if ind == 0:
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return lmlist


def main():
    bTime = 0
    cTime: int = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findpositions(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        cTime = time.time()
        fps = 1 / (cTime - bTime)
        bTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

