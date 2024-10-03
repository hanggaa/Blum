import pyautogui
import cv2
import numpy as np
import keyboard
import time

# Global variables
running = False
x1, y1, x2, y2 = 0, 0, 0, 0

# Load the template image of the Play button
template = cv2.imread('play_button_template.png', 0)
w, h = template.shape[::-1]

def on_mouse(event, x, y, flags, param):
    global x1, y1, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        x2, y2 = x, y
        cv2.destroyAllWindows()

def select_region():
    global x1, y1, x2, y2
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    cv2.namedWindow('Select Region')
    cv2.setMouseCallback('Select Region', on_mouse)
    cv2.imshow('Select Region', screenshot)
    cv2.waitKey(0)
    return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

def detect_objects(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Light blue crystal-like objects
    lower_blue = np.array([90, 100, 200])
    upper_blue = np.array([100, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bright green star-like objects
    lower_green = np.array([50, 200, 200])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Gray bomb objects
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 30, 220])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bomb_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 20
    max_area = 500
    blue_filtered = [cnt for cnt in blue_contours if min_area < cv2.contourArea(cnt) < max_area]
    green_filtered = [cnt for cnt in green_contours if min_area < cv2.contourArea(cnt) < max_area]
    bomb_filtered = [cnt for cnt in bomb_contours if min_area < cv2.contourArea(cnt) < max_area]
    
    return blue_filtered, green_filtered, bomb_filtered

def is_safe_to_click(point, bomb_contours, safety_margin=5):
    for bomb in bomb_contours:
        x, y, w, h = cv2.boundingRect(bomb)
        if x - safety_margin <= point[0] <= x + w + safety_margin and \
           y - safety_margin <= point[1] <= y + h + safety_margin:
            return False
    return True

def detect_play_button(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the match is strong enough, return the center of the matched area
    if max_val > 0.8:  # Adjust this threshold as needed
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    
    return None

def click_objects(region):
    global running
    pyautogui.PAUSE = 0.1  # Slightly increased pause between PyAutoGUI commands
    
    while running:
        if keyboard.is_pressed('q'):
            running = False
            print("Stopping the bot...")
            break

        screenshot = pyautogui.screenshot(region=region)
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Check for Play button
        play_button = detect_play_button(screenshot)
        if play_button:
            pyautogui.click(region[0] + play_button[0], region[1] + play_button[1])
            print(f"Clicked Play button at ({region[0] + play_button[0]}, {region[1] + play_button[1]})")
            time.sleep(0.5)  # Wait for the game to restart
            continue
        
        blue_contours, green_contours, bomb_contours = detect_objects(screenshot)
        
        # Click blue objects first
        for contour in blue_contours:
            if not running:
                break
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if is_safe_to_click((cX, cY), bomb_contours):
                    pyautogui.click(region[0] + cX, region[1] + cY)
                    print(f"Clicked blue object at ({region[0] + cX}, {region[1] + cY})")
                    time.sleep(0.02)  # Small delay between clicks
        
        # Then click green objects
        for contour in green_contours:
            if not running:
                break
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if is_safe_to_click((cX, cY), bomb_contours):
                    pyautogui.click(region[0] + cX, region[1] + cY)
                    print(f"Clicked green object at ({region[0] + cX}, {region[1] + cY})")
                    time.sleep(0.02)  # Small delay between clicks
        
        if keyboard.is_pressed('q'):
            running = False
            print("Stopping the bot...")
            break
        
        time.sleep(0.02)  # Slightly increased delay to reduce CPU usage and slow down the loop

def main():
    global running
    print("Select the game window region...")
    region = select_region()
    print(f"Selected region: {region}")
    
    print("Press 'S' to start the bot, 'Q' to quit.")
    while True:
        if keyboard.is_pressed('s') and not running:
            running = True
            print("Bot started. Press 'Q' to stop.")
            click_objects(region)
        elif keyboard.is_pressed('q'):
            if running:
                running = False
                print("Bot stopped. Press 'S' to start again or 'Q' to quit.")
            else:
                print("Exiting the program.")
                break
        
        time.sleep(0.1)  # Add a small delay to reduce CPU usage

if __name__ == "__main__":
    main()