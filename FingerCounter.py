import cv2
import time
import os
import HandTrackerModule as htm

cap = cv2.VideoCapture(0)

path = "fingers"
overlay_list = []
for i in range(6):
    image = cv2.imread(f'{path}/{i}.jpg')
    image = cv2.resize(image, (200, 200))
    overlay_list.append(image)

pTime = 0
CTime = 0

detector = htm.HandDetector(detectionCon=0.75)

tipsIds = [4, 8, 12, 16, 20]

total_fingers = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        if abs(lmList[4][1] - lmList[2][1]) > 40:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipsIds[id]][2] < lmList[tipsIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

    if len(lmList) != 0 and total_fingers < len(overlay_list):
        h, w, c = overlay_list[total_fingers].shape
        img[0:h, 0:w] = overlay_list[total_fingers]

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, str(int(fps)), (550, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
