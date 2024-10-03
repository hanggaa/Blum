import pyautogui
import cv2
import numpy as np
import keyboard
import time

# Global variables
running = False
x1, y1, x2, y2 = 0, 0, 0, 0

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

def detect_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def click_green_objects(region):
    global running
    while running:
        if keyboard.is_pressed('q'):
            running = False
            print("Stopping the bot...")
            break

        screenshot = pyautogui.screenshot(region=region)
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        contours = detect_green(screenshot)
        
        for contour in contours:
            if not running:
                break
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                pyautogui.click(region[0] + cX, region[1] + cY)
                time.sleep(0.1)  # Add a small delay to avoid clicking too fast
            
            if keyboard.is_pressed('q'):
                running = False
                print("Stopping the bot...")
                break
        
        time.sleep(0.1)  # Add a small delay to reduce CPU usage

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
            click_green_objects(region)
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