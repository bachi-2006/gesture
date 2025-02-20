import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from ctypes import windll
import win32gui
import win32con
import os
import webbrowser  

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Camera setup
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Screen settings
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Cursor control
        self.cursor_speed_multiplier = 1.5
        self.edge_acceleration = 1.05
        self.movement_smoothing = 0.5
        self.prev_x = None
        self.prev_y = None
        
        # Scroll settings
        self.scroll_speed = 200
        self.scroll_smoothing = 0.8
        self.prev_y_position = None
        self.movement_threshold = 0.02
        self.last_scroll_time = time.time()
        self.scroll_cooldown = 0.02
        self.scroll_state = None
        self.scroll_momentum = 0
        self.scroll_momentum_decay = 0.95
        self.continuous_scroll_threshold = 0.15
        
        # Click detection
        self.click_cooldown = 0.15
        self.last_click_time = 0
        self.click_threshold = 0.028  # Same as 2.py
        self.double_click_threshold = 0.3
        self.last_click_type = None
        self.click_count = 0
        
        # Mode tracking
        self.is_scroll_mode = False
        
        # Frame processing
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = 1
        
        # New gesture states
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # Cooldown between gestures
        self.prev_hand_state = None
        
        # Windows specific setup
        self.user32 = windll.user32
        
        # Volume control settings
        self.volume_cooldown = 0.02
        self.last_volume_time = 0
        self.thumb_angle_threshold = 30
        
        # Application launch settings
        self.app_launch_cooldown = 1.0
        self.last_app_launch_time = 0
        self.peace_sign_threshold = 0.1
        self.youtube_url = "https://www.youtube.com"  # Added YouTube URL
        
        # Tab switching settings
        self.tab_switch_threshold = 0.1  # Threshold for horizontal movement
        self.last_tab_switch_time = 0
        self.tab_switch_cooldown = 0.3
        self.is_alt_tab_active = False
        self.prev_hand_x = None
        
        # Task view state
        self.task_view_active = False
        self.task_view_cooldown = 1.0  # 1 second cooldown
        self.last_task_view_time = 0
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # White
        self.bg_color = (0, 0, 0)  # Black
        self.status_text = ""
        self.current_gesture = ""
        self.status_duration = 2.0
        self.last_status_time = 0
        self.overlay_alpha = 0.4  # Transparency for overlay

    def count_extended_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]  
        finger_pips = [6, 10, 14, 18]
        
        extended_fingers = []
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                extended_fingers.append(tip)
        
        return extended_fingers

    def get_hand_state(self, hand_landmarks):
        """Determine the current hand state based on finger positions"""
        try:
            # Finger indices
            thumb = 4
            index = 8
            middle = 12
            ring = 16
            pinky = 20
            
            # Get all finger positions
            thumb_tip = hand_landmarks.landmark[thumb]
            index_tip = hand_landmarks.landmark[index]
            middle_tip = hand_landmarks.landmark[middle]
            ring_tip = hand_landmarks.landmark[ring]
            pinky_tip = hand_landmarks.landmark[pinky]
            
            # Get MCP (knuckle) positions for better reference
            thumb_mcp = hand_landmarks.landmark[2]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            ring_mcp = hand_landmarks.landmark[13]
            pinky_mcp = hand_landmarks.landmark[17]
            
            # Check if fingers are extended by comparing with MCP positions
            fingers_extended = []
            
            # Special check for thumb
            if thumb_tip.x < thumb_mcp.x:  # For right hand
                fingers_extended.append(thumb)
            
            # Check other fingers
            if index_tip.y < index_mcp.y:
                fingers_extended.append(index)
            if middle_tip.y < middle_mcp.y:
                fingers_extended.append(middle)
            if ring_tip.y < ring_mcp.y:
                fingers_extended.append(ring)
            if pinky_tip.y < pinky_mcp.y:
                fingers_extended.append(pinky)
            
            # Get wrist and middle finger base for palm orientation
            wrist = hand_landmarks.landmark[0]
            middle_finger_base = hand_landmarks.landmark[9]
            
            # Calculate palm orientation more accurately
            palm_direction = middle_finger_base.z - wrist.z
            is_palm_backward = palm_direction > 0.1  # Adjusted threshold
            
            # Determine hand state with improved accuracy
            num_extended = len(fingers_extended)
            
            # Check for three fingers (index, middle, ring)
            three_fingers_up = (index in fingers_extended and 
                              middle in fingers_extended and 
                              ring in fingers_extended and 
                              thumb not in fingers_extended and 
                              pinky not in fingers_extended)
            
            # Check for fist (all fingers closed)
            is_fist = num_extended == 0
            
            # Check for open hand (all fingers extended)
            is_open_hand = num_extended >= 4
            
            # Improved thumb up/down detection
            is_thumb_up = (
                thumb_tip.y < thumb_mcp.y - 0.15 and  # Thumb clearly up
                abs(thumb_tip.x - thumb_mcp.x) < 0.15 and  # Not too far sideways
                all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
                    for tip, pip in [(8,6), (12,10), (16,14), (20,18)])  # Other fingers closed
            )
            
            is_thumb_down = (
                thumb_tip.y > thumb_mcp.y + 0.15 and  # Thumb clearly down
                abs(thumb_tip.x - thumb_mcp.x) < 0.15 and  # Not too far sideways
                all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
                    for tip, pip in [(8,6), (12,10), (16,14), (20,18)])  # Other fingers closed
            )
            
            # Peace sign detection (index and middle fingers up, others down)
            is_peace_sign = (
                index_tip.y < index_mcp.y - 0.15 and  # Index up
                middle_tip.y < middle_mcp.y - 0.15 and  # Middle up
                thumb_tip.y > thumb_mcp.y and  # Thumb down
                ring_tip.y > ring_mcp.y and  # Ring down
                pinky_tip.y > pinky_mcp.y and  # Pinky down
                abs(index_tip.y - middle_tip.y) < 0.1  # Index and middle at similar height
            )
            
            # Detect four fingers without thumb
            is_four_fingers = (
                index_tip.y < index_mcp.y - 0.15 and  # Index up
                middle_tip.y < middle_mcp.y - 0.15 and  # Middle up
                ring_tip.y < ring_mcp.y - 0.15 and  # Ring up
                pinky_tip.y < pinky_mcp.y - 0.15 and  # Pinky up
                thumb_tip.y > thumb_mcp.y and  # Thumb down
                abs(index_tip.y - middle_tip.y) < 0.1 and  # Fingers aligned
                abs(middle_tip.y - ring_tip.y) < 0.1 and
                abs(ring_tip.y - pinky_tip.y) < 0.1
            )
            
            # Return states including new gestures
            if is_four_fingers:
                return "FOUR_FINGERS"
            elif is_thumb_up:
                return "THUMBS_UP"
            elif is_thumb_down:
                return "THUMBS_DOWN"
            elif is_peace_sign:
                return "PEACE_SIGN"
            elif is_fist:
                return "FIST"
            elif is_open_hand:
                if is_palm_backward:
                    return "BACK_HAND"
                return "OPEN_HAND"
            elif three_fingers_up:
                return "THREE_FINGERS"
            
            return "OTHER"
            
        except Exception as e:
            print(f"Error in get_hand_state: {e}")
            return "OTHER"

    def show_status(self, text):
        """Update status text to show on camera feed"""
        self.status_text = text
        self.last_status_time = time.time()

    def draw_overlay(self, frame):
        """Draw a beautiful overlay with status and hand tracking info"""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        
        # Draw status bar background
        cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
        
        # Add gesture name
        gesture_text = f"Gesture: {self.current_gesture}"
        cv2.putText(overlay, gesture_text, (20, h-50), 
                   self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Add status text if within duration
        if time.time() - self.last_status_time < self.status_duration:
            # Calculate text size for centering
            text_size = cv2.getTextSize(self.status_text, self.font, self.font_scale, self.font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 25
            
            # Draw operation text
            cv2.putText(overlay, f"Operation: {self.status_text}", (text_x, text_y), 
                       self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)
        
        return frame

    def handle_gestures(self, hand_state, hand_landmarks=None):
        """Handle different gesture states and perform corresponding actions"""
        current_time = time.time()
        
        try:
            # Volume control with rapid response
            if hand_state == "THUMBS_UP" and current_time - self.last_volume_time > self.volume_cooldown:
                self.show_status("Volume Up")
                for _ in range(5):
                    pyautogui.press('volumeup')
                    time.sleep(0.01)
                self.last_volume_time = current_time
                
            elif hand_state == "THUMBS_DOWN" and current_time - self.last_volume_time > self.volume_cooldown:
                self.show_status("Volume Down")
                for _ in range(5):
                    pyautogui.press('volumedown')
                    time.sleep(0.01)
                self.last_volume_time = current_time
                
            # Handle YouTube launch with improved Chrome method
            elif hand_state == "FOUR_FINGERS" and current_time - self.last_app_launch_time > self.app_launch_cooldown:
                self.show_status("Opening YouTube...")
                webbrowser.open(self.youtube_url)
                self.last_app_launch_time = current_time
                
            elif hand_state == "PEACE_SIGN" and current_time - self.last_app_launch_time > self.app_launch_cooldown:
                self.show_status("Opening Chrome...")
                try:
                    webbrowser.open("https://www.google.com/chrome/")
                except:
                    try:
                        webbrowser.open("https://www.google.com/chrome/")
                        time.sleep(1)
                        pyautogui.hotkey('ctrl', 'l')
                        time.sleep(0.2)
                        pyautogui.write('youtube.com')
                        time.sleep(0.2)
                        pyautogui.press('enter')
                    except Exception as e:
                        self.show_status(f"Failed to open Chrome")
                
                self.last_app_launch_time = current_time
                
            # Handle window and other gestures
            elif hand_state == "FIST" and current_time - self.last_gesture_time > self.gesture_cooldown:
                self.show_status("Minimizing Window")
                window = win32gui.GetForegroundWindow()
                if window:
                    win32gui.ShowWindow(window, win32con.SW_MINIMIZE)
                    self.last_gesture_time = current_time
                    
            elif hand_state == "OPEN_HAND" and self.prev_hand_state == "FIST" and current_time - self.last_gesture_time > self.gesture_cooldown:
                self.show_status("Maximizing Window")
                window = win32gui.GetForegroundWindow()
                if window:
                    placement = win32gui.GetWindowPlacement(window)
                    if placement[1] == win32con.SW_SHOWMINIMIZED:
                        win32gui.ShowWindow(window, win32con.SW_RESTORE)
                    else:
                        win32gui.ShowWindow(window, win32con.SW_MAXIMIZE)
                    self.last_gesture_time = current_time
                    
            elif hand_state == "BACK_HAND" and current_time - self.last_gesture_time > self.gesture_cooldown:
                self.show_status("Back")
                pyautogui.hotkey('alt', 'left')
                self.last_gesture_time = current_time
                
            elif hand_state == "THREE_FINGERS" and current_time - self.last_gesture_time > self.gesture_cooldown:
                # Only activate task view if it's not already active and cooldown has passed
                if not self.task_view_active and current_time - self.last_task_view_time > self.task_view_cooldown:
                    self.show_status("Task View")
                    pyautogui.keyDown('win')
                    time.sleep(0.1)
                    pyautogui.press('tab')
                    time.sleep(0.1)
                    pyautogui.keyUp('win')
                    self.task_view_active = True
                    self.last_task_view_time = current_time
                    self.last_gesture_time = current_time
            
            # Reset task view state when hand state changes
            elif self.prev_hand_state == "THREE_FINGERS" and hand_state != "THREE_FINGERS":
                self.task_view_active = False
            
            # Release alt key if not in four fingers mode
            if hand_state != "FOUR_FINGERS" and self.is_alt_tab_active:
                pyautogui.keyUp('alt')
                self.is_alt_tab_active = False
                self.prev_hand_x = None
            
            self.prev_hand_state = hand_state
            
        except Exception as e:
            self.show_status(f"Error: {str(e)}")
            if self.is_alt_tab_active:
                pyautogui.keyUp('alt')
                self.is_alt_tab_active = False

    def process_landmarks(self, hand_landmarks, frame_width, frame_height):
        try:
            # Get current hand state
            hand_state = self.get_hand_state(hand_landmarks)
            self.current_gesture = hand_state
            # Pass hand_landmarks to handle_gestures for tab switching
            self.handle_gestures(hand_state, hand_landmarks)
            
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            current_time = time.time()
            
            extended_fingers = self.count_extended_fingers(hand_landmarks)
            self.is_scroll_mode = len(extended_fingers) >= 4
            
            if self.is_scroll_mode:
                if self.prev_y_position is not None:
                    y_movement = palm_center.y - self.prev_y_position
                    
                    if abs(y_movement) > self.movement_threshold:
                        self.scroll_state = "up" if y_movement < 0 else "down"
                        self.scroll_momentum = abs(y_movement / self.movement_threshold)
                    elif palm_center.y > (1 - self.continuous_scroll_threshold):
                        self.scroll_state = "down"
                        self.scroll_momentum = 1.0
                    elif palm_center.y < self.continuous_scroll_threshold:
                        self.scroll_state = "up"
                        self.scroll_momentum = 1.0
                    
                    if self.scroll_state and current_time - self.last_scroll_time > self.scroll_cooldown:
                        scroll_amount = int(self.scroll_speed * self.scroll_momentum)
                        if self.scroll_state == "up":
                            pyautogui.scroll(scroll_amount)
                        else:
                            pyautogui.scroll(-scroll_amount)
                        self.last_scroll_time = current_time
                        self.last_click_type = f"Scroll {self.scroll_state.title()}"
                        
                        if not (palm_center.y > (1 - self.continuous_scroll_threshold) or 
                               palm_center.y < self.continuous_scroll_threshold):
                            self.scroll_momentum *= self.scroll_momentum_decay
                
                self.prev_y_position = palm_center.y
                return None, None  # Don't move cursor in scroll mode
            
            # Cursor movement
            normalized_x = (index_tip.x * 1.1) - 0.05
            normalized_y = (index_tip.y * 1.1) - 0.05
            
            normalized_x = max(0, min(1, normalized_x))
            normalized_y = max(0, min(1, normalized_y))
            
            if 0.1 < normalized_x < 0.9 and 0.1 < normalized_y < 0.9:
                edge_mult = 1.0
            else:
                edge_mult = self.edge_acceleration
            
            target_x = int(normalized_x * self.screen_width)
            target_y = int(normalized_y * self.screen_height)
            
            if self.prev_x is not None and self.prev_y is not None:
                x = int(self.prev_x + (target_x - self.prev_x) * self.movement_smoothing)
                y = int(self.prev_y + (target_y - self.prev_y) * self.movement_smoothing)
            else:
                x = target_x
                y = target_y
            
            # Apply cursor speed and edge acceleration
            dx = x - (self.prev_x or x)
            dy = y - (self.prev_y or y)
            x = int(x + dx * (self.cursor_speed_multiplier - 1) * edge_mult)
            y = int(y + dy * (self.cursor_speed_multiplier - 1) * edge_mult)
            
            # Keep cursor within screen bounds
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            
            # Click detection
            thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                                  (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            
            if thumb_index_distance < self.click_threshold:
                if current_time - self.last_click_time < self.double_click_threshold:
                    if self.click_count == 1:
                        pyautogui.doubleClick()
                        self.click_count = 0
                        self.last_click_type = "Double Click"
                        self.show_status("Double Click")
                    else:
                        self.click_count += 1
                else:
                    pyautogui.click()
                    self.click_count = 1
                    self.last_click_type = "Click"
                    self.show_status("Click")
                self.last_click_time = current_time
            
            if current_time - self.last_click_time > self.double_click_threshold:
                self.click_count = 0
            
            self.prev_x = x
            self.prev_y = y
            
            return x, y
            
        except Exception as e:
            self.show_status(f"Error: {str(e)}")
            return None, None

    def process_frame(self, frame):
        """Process each frame from the camera"""
        try:
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Skip frames to improve performance
            self.frame_count += 1
            if self.frame_count % self.PROCESS_EVERY_N_FRAMES != 0:
                return frame, None
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(rgb_frame)
            
            cursor_pos = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Process landmarks for cursor control and gestures
                    cursor_pos = self.process_landmarks(hand_landmarks, frame_width, frame_height)
            else:
                # Reset states when no hand is detected
                self.prev_x = None
                self.prev_y = None
                self.prev_hand_state = None
                self.scroll_state = None
                self.scroll_momentum = 0
                
                if self.is_alt_tab_active:
                    pyautogui.keyUp('alt')
                    self.is_alt_tab_active = False
            
            # Draw overlay with status
            frame = self.draw_overlay(frame)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            return frame, cursor_pos
            
        except Exception as e:
            self.show_status(f"Error: {str(e)}")
            return frame, None

def main():
    try:
        recognizer = HandGestureRecognizer()
        
        while True:
            ret, frame = recognizer.cap.read()
            if not ret:
                break
                
            frame, cursor_pos = recognizer.process_frame(frame)
            if cursor_pos:
                x, y = cursor_pos
                if x is not None and y is not None:
                    pyautogui.moveTo(x, y)
            
            if cv2.waitKey(1) & 0xFF == 27:  
                break
                
        recognizer.cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in main: {e}")
        
if __name__ == "__main__":
    main()
