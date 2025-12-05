import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from ctypes import windll
import win32gui
import win32con
import webbrowser

class HandGestureRecognizer:
    def __init__(self):

        # HAND SETUP
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # CAMERA
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # SCREEN
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

        # CURSOR
        self.cursor_smooth = 0.6
        self.prev_x = None
        self.prev_y = None

        # CLICK
        self.last_click = 0
        self.click_cooldown = 0.32
        self.click_distance = 0.03

        # --------------------------
        # ⭐ GLOBAL LATENCY CONTROL
        # --------------------------
        self.global_delay = 0.35   # Applied to every gesture!

        # --------------------------
        # ⭐ INDIVIDUAL GESTURE DELAY
        # --------------------------
        self.latency = {
            "THUMB_UP": 0.15,
            "THUMB_DOWN": 0.15,
            "FIST": 0.5,
            "OPEN": 0.6,
            "FOUR_FINGERS": 1.3,   # slow because it opens Youtube
            "PEACE_SIGN": 1.3      # slow because it opens Chrome
        }

        # STATE
        self.last_gesture_state = ""
        self.last_gesture_time = 0

        # STATUS
        self.status_text = ""
        self.last_status_time = 0
        self.status_duration = 1.5
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # -------------------------------------------------------
    def gesture_ready(self, gesture):
        """
        Checks if enough latency/delay has passed
        before allowing the gesture again.
        """

        now = time.time()

        # Determine delay:
        delay = self.latency.get(gesture, self.global_delay)

        # Gesture changed → allow immediately
        if gesture != self.last_gesture_state:
            self.last_gesture_state = gesture
            self.last_gesture_time = now
            return True

        # Same gesture → wait required delay
        if now - self.last_gesture_time < delay:
            return False

        self.last_gesture_time = now
        return True

    # -------------------------------------------------------
    def show_status(self, msg):
        self.status_text = msg
        self.last_status_time = time.time()

    # -------------------------------------------------------
    def get_hand_state(self, lm):
        try:
            wrist = lm[0]
            index_tip = lm[8]
            middle_tip = lm[12]
            ring_tip = lm[16]
            pinky_tip = lm[20]
            thumb_tip = lm[4]

            index_mcp = lm[5]
            middle_mcp = lm[9]
            ring_mcp = lm[13]
            pinky_mcp = lm[17]
            thumb_mcp = lm[2]

            # Finger extension
            idx = index_tip.y < index_mcp.y - 0.1
            mid = middle_tip.y < middle_mcp.y - 0.1
            rng = ring_tip.y < ring_mcp.y - 0.1
            pnk = pinky_tip.y < pinky_mcp.y - 0.1

            thumb_up = thumb_tip.y < thumb_mcp.y - 0.15
            thumb_down = thumb_tip.y > thumb_mcp.y + 0.15

            count = sum([idx, mid, rng, pnk])

            if count == 0 and not thumb_up and not thumb_down:
                return "FIST"

            if count == 4:
                return "FOUR_FINGERS"

            if count == 2 and idx and mid:
                return "PEACE_SIGN"

            if thumb_up and not idx:
                return "THUMB_UP"

            if thumb_down and not idx:
                return "THUMB_DOWN"

            if count == 5:
                return "OPEN"

            return "OTHER"

        except:
            return "OTHER"

    # -------------------------------------------------------
    def perform_actions(self, gesture):

        # Prevent repeated triggers by latency control
        if not self.gesture_ready(gesture):
            return

        # Volume Up
        if gesture == "THUMB_UP":
            self.show_status("Volume Up")
            pyautogui.press("volumeup")

        # Volume Down
        elif gesture == "THUMB_DOWN":
            self.show_status("Volume Down")
            pyautogui.press("volumedown")

        # Minimize
        elif gesture == "FIST":
            self.show_status("Minimize")
            win32gui.ShowWindow(win32gui.GetForegroundWindow(), win32con.SW_MINIMIZE)

        # Maximize
        elif gesture == "OPEN":
            self.show_status("Maximize")
            win32gui.ShowWindow(win32gui.GetForegroundWindow(), win32con.SW_MAXIMIZE)

        # Youtube
        elif gesture == "FOUR_FINGERS":
            self.show_status("Opening YouTube…")
            webbrowser.open("https://youtube.com")

        # Chrome
        elif gesture == "PEACE_SIGN":
            self.show_status("Opening Chrome…")
            webbrowser.open("https://google.com")

    # -------------------------------------------------------
    def process_landmarks(self, lm):
        index_tip = lm[8]
        thumb_tip = lm[4]

        # Cursor
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)

        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        smooth_x = int(self.prev_x + (x - self.prev_x) * self.cursor_smooth)
        smooth_y = int(self.prev_y + (y - self.prev_y) * self.cursor_smooth)

        pyautogui.moveTo(smooth_x, smooth_y)

        self.prev_x, self.prev_y = smooth_x, smooth_y

        # Click
        dist = np.hypot(
            index_tip.x - thumb_tip.x,
            index_tip.y - thumb_tip.y
        )

        if dist < self.click_distance and time.time() - self.last_click > self.click_cooldown:
            pyautogui.click()
            self.show_status("Click")
            self.last_click = time.time()

    # -------------------------------------------------------
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark

                self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

                gesture = self.get_hand_state(lm)
                self.perform_actions(gesture)

                self.process_landmarks(lm)

            # Status text
            if time.time() - self.last_status_time < self.status_duration:
                cv2.putText(frame, self.status_text, (20, 440),
                            self.font, 1, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


# RUN
if __name__ == "__main__":
    HandGestureRecognizer().run()
