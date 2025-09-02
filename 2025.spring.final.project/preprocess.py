# OCR之前的图像预处理
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# 增强对比度
def contrast(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12,12))
    return clahe.apply(img)

if __name__ == "__main__":
    with open('input/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    os.makedirs("input/preprocessed_test_images", exist_ok=True)
    for item in test_data:
        filename = item.get("path", "")
        image_path = os.path.join("input/test_images", filename)
        if not os.path.exists(image_path):
            print(f"{image_path} No exist!")
            continue
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度
        img1 = contrast(gray)
        processed_path = os.path.join("input/preprocessed_test_images", filename)
        cv2.imwrite(processed_path, img1)

    print("preprocess finished.")