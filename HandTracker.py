import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0

while True:
    #our frame
    success, img = capture.read()
    img = cv2.flip(img, 1)  # Usunięcie odbicia lustrzanego
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f'id: {id}, cx: {cx}, cy: {cy}')
                #Czubki palców
                # if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                #     cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

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