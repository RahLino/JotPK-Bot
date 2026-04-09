import cv2
import numpy as np
import mss
import pydirectinput
import time
from ultralytics import YOLO

# Configuration: Define the screen area of the mini-game
# You can use a tool like 'PowerToys' or simple print-screen to find these coords
GAME_AREA = {"top": 100, "left": 100, "width": 600, "height": 600}

# Load YOLO model. For the real game, you would want to train a custom YOLO model
# on screenshots of Journey of the Prairie King and use it here (e.g., 'prairie_king_model.pt')
# For now, we use a placeholder model yolov8n.pt
model = YOLO('yolov8n.pt')

# Assume custom model classes: 0 -> Player, 1 -> Enemy
PLAYER_CLASS_ID = 0
ENEMY_CLASS_ID = 1

def process_frame(sct):
    # Capture the screen
    img = np.array(sct.grab(GAME_AREA))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def main():
    with mss.mss() as sct:
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
