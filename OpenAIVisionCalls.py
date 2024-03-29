import base64
import cv2 as cv
import time
import os
import shutil
import threading
from openai import OpenAI
from OpenAIApiCalls import OpenAPICalls

class OpenVisionCalls(OpenAPICalls):
    def __init__(self):
        super().__init__()

    def capture_frames(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            current_time = time.time()
            if current_time - start_time >= 5:
                start_time = current_time
                image_file = f"image_{int(start_time)}.jpg"
                cv.imwrite(image_file, frame)
                self.send_to_vision(image_file)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    def send_to_vision(self, image_file):
        with open(image_file, "rb") as image:
            image_data = base64.b64encode(image.read()).decode("utf-8")
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what is in this image."},
                        {"type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{image_data}"
                         },
                         },
                    ],
                }],
                max_tokens=300,
            )
        target_directory = "images"
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        target_file = os.path.join(target_directory, os.path.basename(image_file))

        try:
            shutil.move(image_file, target_file)
            #print(f"Image '{image_file}' moved to '{target_file}' successfully.")
        except Exception as e:
            print(f"Error moving image '{image_file}' to '{target_file}': {e}")

        with open("./conversation_text/vision_data.txt", "a") as vision:
            vision_data = response.choices[0].message.content
            vision.write(vision_data + "\n")

def run_vision_capture():
    vision_calls = OpenVisionCalls()
    vision_calls.capture_frames()
