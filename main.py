import cv2
import numpy as np
import mss
import pydirectinput
import time
import pygetwindow as gw
from ultralytics import YOLO

# Configuration: Define the screen area of the mini-game
# You can use a tool like 'PowerToys' or simple print-screen to find these coords
GAME_AREA = {"top": 100, "left": 100, "width": 600, "height": 600}

def update_game_area():
    global GAME_AREA
    windows = gw.getWindowsWithTitle("Stardew Valley")
    if windows:
        window = windows[0]
        GAME_AREA["top"] = window.top
        GAME_AREA["left"] = window.left
        GAME_AREA["width"] = window.width
        GAME_AREA["height"] = window.height
        print(f"Found Stardew Valley window: {GAME_AREA}")
    else:
        print("Stardew Valley window not found. Using default GAME_AREA.")

# Load YOLO model. For the real game, you would want to train a custom YOLO model
# on screenshots of Journey of the Prairie King and use it here (e.g., 'prairie_king_model.pt')
# For now, we use a placeholder model yolov8n.pt
model = YOLO('yolov8n.pt')

# Assume custom model classes: 0 -> Player, 1 -> Enemy
PLAYER_CLASS_ID = 0
ENEMY_CLASS_ID = 1

current_keys = set()

def update_keys(desired_keys):
    global current_keys

    # Keys to release
    for key in current_keys - desired_keys:
        pydirectinput.keyUp(key)

    # Keys to press
    for key in desired_keys - current_keys:
        pydirectinput.keyDown(key)

    current_keys = desired_keys

def process_frame(sct):
    # Capture the screen
    img = np.array(sct.grab(GAME_AREA))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def main():
    update_game_area()
    with mss.mss() as sct:
        try:
            while True:
                frame = process_frame(sct)

                # Run YOLO inference
                results = model(frame, verbose=False)

                player_pos = []
                enemies = []

                # Process results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get class ID
                        cls_id = int(box.cls[0].item())
                        # Get center coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        if cls_id == PLAYER_CLASS_ID:
                            player_pos.append((cx, cy))
                        elif cls_id == ENEMY_CLASS_ID:
                            enemies.append((cx, cy))
                
                desired_keys = set()
                
                if player_pos and enemies:
                    px, py = player_pos[0]

                    # LOGIC: Find nearest enemy
                    nearest_enemy = min(enemies, key=lambda e: np.linalg.norm(np.array([px, py]) - np.array(e)))
                    ex, ey = nearest_enemy

                    # ACTION: Shoot at nearest enemy
                    # Simple logic: if enemy is to the right, shoot right
                    if ex > px + 10: desired_keys.add('right')
                    elif ex < px - 10: desired_keys.add('left')

                    if ey > py + 10: desired_keys.add('down')
                    elif ey < py - 10: desired_keys.add('up')

                    # ACTION: Movement (Kiting)
                    # Move away from the nearest enemy
                    if ex > px: desired_keys.add('a')
                    elif ex < px: desired_keys.add('d')

                    if ey > py: desired_keys.add('w')
                    elif ey < py - 10: desired_keys.add('s')

                update_keys(desired_keys)
                
                # Press 'q' to quit bot
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Ensure keys are released when the bot stops or crashes
            update_keys(set())

if __name__ == "__main__":
    # Give yourself 5 seconds to switch to the game window
    time.sleep(5)
    main()
