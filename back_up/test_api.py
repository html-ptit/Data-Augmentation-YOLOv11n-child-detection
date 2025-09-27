import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("API_URL")

image_path = r"C:\linh tinh 1\daycare.v2i.yolov8\test\images\resized-scene00181_png.rf.bb5397d8640f29e7a66ac3f5633fa371.jpg"

try:
    with open(image_path, "rb") as img:
        files = {
            "file": (os.path.basename(image_path), img, "image/jpeg")
        }
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.post(url, files=files, headers=headers)
        with open("response.json", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Status:", response.status_code)
        print("Response:", response.text)

except Exception as e:
    print("Error:", e)
