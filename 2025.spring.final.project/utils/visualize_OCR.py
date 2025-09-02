import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

def visualize_char_boxes(image_path, boxes, 
                         box_color='red', line_thickness=1):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"{image_path} loaded error.")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换为RGB格式(用于matplotlib显示)
    img_with_boxes = img_rgb.copy()
    plt.figure(figsize=(12, 12))
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'purple': (255, 0, 255),
        'cyan': (255, 255, 0),
    }
    box_color_bgr = color_map.get(box_color.lower(), (0, 0, 255))  # 默认红色
    
    # 在图像上绘制边界框和文本
    for item in boxes:
        x1, y1, x2, y2 = item['box']['start_x'], item['box']['start_y'], item['box']['end_x'],  item['box']['end_y']
        # 绘制矩形边界框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), 
                     box_color_bgr, line_thickness)
        
    # 使用matplotlib显示图像
    plt.imshow(img_with_boxes)
    plt.title('OCR visualization with bounding boxes')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return img_with_boxes

image_path = "input/preprocessed_test_images/2110.jpg"
with open('input/ocr_test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

result = next((item['char_bounding_box_list'] for item in test_data if item['fk_homework_id'] == 2110), None)
visualize_char_boxes(
    image_path=image_path,
    boxes = result,
    box_color='red'
)
