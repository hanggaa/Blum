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
    lower_gray = np.array([0, 0, 60])  # Lowered the lower bound
    upper_gray = np.array([180, 60, 255])  # Increased the upper bound
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

def is_safe_to_click(point, bomb_contours, safety_margin=20):  # Increased safety margin
    for bomb in bomb_contours:
        x, y, w, h = cv2.boundingRect(bomb)
        if x - safety_margin <= point[0] <= x + w + safety_margin and \
           y - safety_margin <= point[1] <= y + h + safety_margin:
            return False
    return True

def detect_play_button(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Focus on a small area around where we expect the button to be
    height, width = gray.shape
    roi_y = int(height * 0.8)  # Start at 80% of the height
    roi_height = height - roi_y
    roi_x = int(width * 0.3)  # Start at 30% of the width
    roi_width = int(width * 0.4)  # Use 40% of the width
    
    roi = gray[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    
    # Perform template matching
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the match is strong enough, return True
    if max_val > 0.8:  # Adjust this threshold as needed
        return True
    
    return False

def click_objects(region):
    global running
    pyautogui.PAUSE = 0.1

    # Calculate the center of the Play button based on the observed clicks
    play_button_x = 912 - region[0]  # Relative x-coordinate within the region
    play_button_y = 630 - region[1]  # Relative y-coordinate within the region

    while running:
        if keyboard.is_pressed('q'):
            running = False
            print("Stopping the bot...")
            break

        screenshot = pyautogui.screenshot(region=region)
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Check for Play button
        if detect_play_button(screenshot):
            # Use the pre-calculated Play button coordinates
            click_x = region[0] + play_button_x
            click_y = region[1] + play_button_y
            
            pyautogui.click(click_x, click_y)
            print(f"Clicked Play button at ({click_x}, {click_y})")
            time.sleep(0.5)  # Wait for the game to restart
            continue
        
        blue_contours, green_contours, bomb_contours = detect_objects(screenshot)
        
        # New function to check if a point is far enough from all bombs
        def is_far_from_bombs(point, bomb_contours, min_distance=20):
            for bomb in bomb_contours:
                M = cv2.moments(bomb)
                if M["m00"] != 0:
                    bX = int(M["m10"] / M["m00"])
                    bY = int(M["m01"] / M["m00"])
                    if ((point[0] - bX) ** 2 + (point[1] - bY) ** 2) ** 0.5 < min_distance:
                        return False
            return True

        # Click blue objects first
        for contour in blue_contours:
            if not running:
                break
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if is_safe_to_click((cX, cY), bomb_contours) and is_far_from_bombs((cX, cY), bomb_contours):
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
                if is_safe_to_click((cX, cY), bomb_contours) and is_far_from_bombs((cX, cY), bomb_contours):
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