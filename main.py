import cv2
import numpy as np
import mss
import pydirectinput
import time

# Configuration: Define the screen area of the mini-game
# You can use a tool like 'PowerToys' or simple print-screen to find these coords
GAME_AREA = {"top": 100, "left": 100, "width": 600, "height": 600}

def process_frame(sct):
    # Capture the screen
    img = np.array(sct.grab(GAME_AREA))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def find_objects(img, lower_color, upper_color):
    # Create a mask for the specific color
    mask = cv2.inRange(img, np.array(lower_color), np.array(upper_color))
    # Find contours (blobs of that color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    positions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # Filter small noise
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                positions.append((cX, cY))
    return positions

def main():
    with mss.mss() as sct:
        while True:
            frame = process_frame(sct)
            
            # 1. Find Player (Example Color: Cowboy Hat Brown)
            # You must tune these RGB values!
            player_pos = find_objects(frame, [100, 50, 50], [130, 80, 80])
            
            # 2. Find Enemies (Example Color: Orc Green)
            enemies = find_objects(frame, [50, 100, 50], [80, 150, 80])
            
            if player_pos and enemies:
                px, py = player_pos[0]
                
                # LOGIC: Find nearest enemy
                nearest_enemy = min(enemies, key=lambda e: np.linalg.norm(np.array([px, py]) - np.array(e)))
                ex, ey = nearest_enemy
                
                # ACTION: Shoot at nearest enemy
                # Simple logic: if enemy is to the right, shoot right
                if ex > px + 10: pydirectinput.press('right')
                elif ex < px - 10: pydirectinput.press('left')
                elif ey > py + 10: pydirectinput.press('down')
                elif ey < py - 10: pydirectinput.press('up')
                
                # ACTION: Movement (Kiting)
                # This needs to be independent of shooting in a real thread or async loop
                # for twin-stick feel.
            
            # Press 'q' to quit bot
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    # Give yourself 5 seconds to switch to the game window
    time.sleep(5)
    main()
