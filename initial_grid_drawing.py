import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from collections import deque
import vgamepad as vg
import time

def find_closest_object(grid_layout, start_pos, target_class):
    rows, cols = len(grid_layout), len(grid_layout[0])
    visited = set()
    queue = deque()
    queue.append((start_pos, []))

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) in visited or not (0 <= y < rows and 0 <= x < cols):
            continue
        visited.add((x, y))

        if grid_layout[y][x] == target_class:
            return path

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            queue.append(((nx, ny), path + [(nx, ny)]))

    return None

def move_player_with_gamepad(path, delay=0.3):
    gamepad = vg.VX360Gamepad()

    time.sleep(4)

    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        print(f"Ruch: dx={dx}, dy={dy}")  # debug

        if dx == 1:
            gamepad.left_joystick(x_value=32767, y_value=0)  # right
        elif dx == -1:
            gamepad.left_joystick(x_value=-32767, y_value=0)  # left
        elif dy == 1:
            gamepad.left_joystick(x_value=0, y_value=-32767)  # down
        elif dy == -1:
            gamepad.left_joystick(x_value=0, y_value=32767)  # up
        else:
            gamepad.left_joystick(x_value=0, y_value=0)  # neutral

        gamepad.update()
        time.sleep(delay)

        # Reset pada
        gamepad.left_joystick(x_value=0, y_value=0)
        gamepad.update()
        time.sleep(0.1)


    gamepad.left_joystick(x_value=0, y_value=0)
    gamepad.update()

    # Uzycie narzedzia
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    gamepad.update()
    time.sleep(0.5)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    gamepad.update()

def detect_and_visualize_grid(image_path, model_path='best.pt', tile_size=64, x_offset=0, y_offset=0):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.5)
    result = results[0]

    boxes = result.boxes.xywh.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    names = result.names

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    grid_rows = img_height // tile_size
    grid_cols = img_width // tile_size

    grid_layout = [['empty' for _ in range(grid_cols)] for _ in range(grid_rows)]

    # Hard coded pozycja gracza
    player_pos = (14, 7)

    for box, cls_id in zip(boxes, classes):
        x_center, y_center, w, h = box
        grid_x = int((x_center - x_offset) / tile_size)
        grid_y = int((y_center - y_offset) / tile_size)
        class_name = names[int(cls_id)]

        if 0 <= grid_x < grid_cols and 0 <= grid_y < grid_rows:
            grid_layout[grid_y][grid_x] = class_name

    target_class = "Stone"
    path = find_closest_object(grid_layout, player_pos, target_class)
    if path:
        print("Ścieżka do najbliższego obiektu:", path)
        move_player_with_gamepad([player_pos] + path)
    else:
        print("Nie znaleziono obiektu typu:", target_class)

    # Wizualizacja
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    for y in range(y_offset, img_height + y_offset, tile_size):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    for x in range(x_offset, img_width + x_offset, tile_size):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)

    for y in range(grid_rows):
        for x in range(grid_cols):
            obj = grid_layout[y][x]
            if obj != 'empty':
                text_x = x * tile_size + tile_size // 4 + x_offset
                text_y = y * tile_size + tile_size // 2 + y_offset
                ax.text(text_x, text_y, obj, color='red', fontsize=8, weight='bold')

    if path:
        for (x, y) in path:
            rect = plt.Rectangle((x*tile_size + x_offset, y*tile_size + y_offset), tile_size, tile_size,
                                 linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

    ax.set_title("Tile Grid + sterowanie gamepadem")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    return grid_layout

# Użycie
grid = detect_and_visualize_grid(
    "C:/Users/jedma/PycharmProjects/stardew_bot/test_image.png",
    x_offset=32,
    y_offset=58
)
