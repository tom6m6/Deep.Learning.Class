import base64
import requests
import time
import json
import os
# 百度官方实现:https://ai.baidu.com/ai-doc/OCR/hk3h7y2qq
API_KEY = "***"  # 替换为你的API Key
SECRET_KEY = "***" # 替换为你的Secret Key
access_token_cache = None

def get_access_token():
    global access_token_cache
    if access_token_cache:
        return access_token_cache
    auth_url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials", 
        "client_id": API_KEY, 
        "client_secret": SECRET_KEY
    }
    response = requests.post(auth_url, params = params)

    if response.status_code == 200:
        access_token_cache = response.json().get("access_token")
        return access_token_cache
    else:
        raise Exception(f"Get Access Token Error: {response.text}")
    
def BaiduOCR(image_path):
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode("utf8")
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    params = {
        "image": img,
        "recognize_granularity": "small",  # 定位单字符位置
    }
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token={get_access_token()}"

    while True:
        try:
            response = requests.post(
                url,
                headers = headers,
                data = params,
                timeout = 30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error_code" in result:
                    print(f"API Error: {result['error_msg']}")
                    return None

                source_text = ""
                bounding_box_list = []

                if "words_result" in result:
                    for words in result["words_result"]:
                        if "chars" in words:
                            for c in words["chars"]:
                                location = c["location"]
                                bounding_box_list.append({
                                    "char": c["char"],
                                    "box": {
                                        'start_x': location["left"],
                                        'start_y': location["top"],
                                        'end_x': location["left"] + location["width"],
                                        'end_y': location["top"] + location["height"]
                                    }
                                })
                
                source_text = "".join([c["char"] for c in bounding_box_list])
                return source_text, bounding_box_list
            elif response.status_code == 429:
                print("Too many requests, rest a few seconds.")
                time.sleep(10)
                continue
            else:
                print(f"Request Failed: {response.text}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None

if __name__ == "__main__":
    ocr_data = []
    with open('input/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    for item in test_data:
        update_item = {
            "fk_homework_id": item["fk_homework_id"],
            "path": item["path"],
            "predict_text": "",
            "bounding_box_list": []
        }
        filename = item.get("path", "")
        image_path = os.path.join("input/preprocessed_test_images", filename)
        if not os.path.exists(image_path):
            print(f"image doesn't exist: {image_path}")
            continue
        
        source_text, bounding_box_list = BaiduOCR(image_path)
        if source_text:
            update_item["source_text"] = source_text
            update_item["char_bounding_box_list"] = bounding_box_list
        else:
            print(f"{image_path} failed.")
        ocr_data.append(update_item)

    with open('input/ocr_test_data.json', 'w', encoding='utf-8') as f:
        json.dump(ocr_data, f, ensure_ascii = False, indent = 2)
    
    print("OCR finished.")