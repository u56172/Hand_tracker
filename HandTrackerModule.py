import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                #Czubki palców
                # if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                #     cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                if draw and id == 0:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    capture = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        #our frame
        success, img = capture.read()
        img = cv2.flip(img, 1)  # Usunięcie odbicia lustrzanego
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(f'id: {lmList[0][0]}, cx: {lmList[0][1]}, cy: {lmList[0][2]}')

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        # Zamknięcie przez przycisk X w oknie
        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
