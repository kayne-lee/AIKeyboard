import cv2
import mediapipe as mp
import math
from time import sleep

# Define the keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

# Button class
class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

# Function to draw all buttons with transparent backgrounds and change color if touched
def drawAll(img, buttonList, lmList, detector):
    overlay = img.copy()
    alpha = 0.5  # Transparency factor
    global finalText

    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        color = (255, 0, 0)  # Default color

        # Check if the index finger (landmark 8) is within the button's bounding box
        if lmList and len(lmList) >= 9:
            index_finger_x, index_finger_y = lmList[8][1], lmList[8][2]
            if x < index_finger_x < x + w and y < index_finger_y < y + h:
                color = (0, 255, 0)  # Change color to green
                l, _, _, _ = detector.findDistance(lmList[8], lmList[12], img, draw=True)

                if l < 30:
                    color = (0, 0, 255)
                    finalText += button.text
                    sleep(0.15)

        # Draw the transparent rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(overlay, button.text, (x + 15, y + 55), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Add the overlay
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw the final text area
    cv2.rectangle(img, (50, 350), (700, 450), (0, 0, 128), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    return img

# HandDetector class to detect hands and landmarks
class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize mediapipe hands
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
            h, w, _ = img.shape
            xList, yList = [], []
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return lmList

    def findDistance(self, p1, p2, img=None, draw=True):
        """Finds the Euclidean distance between two points on the hand.

        Args:
            p1 (int): Index of the first landmark.
            p2 (int): Index of the second landmark.
            img (np.ndarray, optional): Image on which to draw. Defaults to None.
            draw (bool, optional): Draws line and circles if True. Defaults to True.

        Returns:
            tuple: (distance, (x1, y1), (x2, y2), center (cx, cy))
        """
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        distance = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        return distance, (x1, y1), (x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2)

# Main function for testing
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8, maxHands=2)
    buttonList = []
    for i in range(len(keys)):
        for x, key in enumerate(keys[i]):
            buttonList.append(Button([100 * x + 50, 100 * i + 50], key))

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)
        img = drawAll(img, buttonList, lmList, detector)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
