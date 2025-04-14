import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

def detect_and_visualize_grid(image_path, model_path='best.pt', tile_size=64, x_offset=0, y_offset=0):
    # 1. Wczytaj model YOLOv8 i wykonaj detekcję
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.5)
    result = results[0]

    # 2. Wyciągnij dane z wyników detekcji
    boxes = result.boxes.xywh.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    names = result.names

    # 3. Wczytaj oryginalny obraz
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    # 4. Ustal wymiary gridu
    grid_rows = img_height // tile_size
    grid_cols = img_width // tile_size

    # 5. Przygotuj pustą mapę gridu
    grid_layout = [['empty' for _ in range(grid_cols)] for _ in range(grid_rows)]

    # 6. Wypełnij grid na podstawie detekcji
    for box, cls_id in zip(boxes, classes):
        x_center, y_center, w, h = box
        grid_x = int((x_center - x_offset) / tile_size)
        grid_y = int((y_center - y_offset) / tile_size)
        class_name = names[int(cls_id)]

        if 0 <= grid_x < grid_cols and 0 <= grid_y < grid_rows:
            grid_layout[grid_y][grid_x] = class_name

    # 7. Wizualizacja
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Rysuj siatkę z offsetem
    for y in range(y_offset, img_height + y_offset, tile_size):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    for x in range(x_offset, img_width + x_offset, tile_size):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)

    # Rysuj nazwy obiektów
    for y in range(grid_rows):
        for x in range(grid_cols):
            obj = grid_layout[y][x]
            if obj != 'empty':
                text_x = x * tile_size + tile_size // 4 + x_offset
                text_y = y * tile_size + tile_size // 2 + y_offset
                ax.text(text_x, text_y, obj, color='red', fontsize=8, weight='bold')

    ax.set_title("Tile Grid z obiektami YOLOv8")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    return grid_layout

# Przykład użycia z offsetem
grid = detect_and_visualize_grid("C:/Users/jedma/PycharmProjects/stardew_bot/test_image.png", x_offset=32, y_offset=58)