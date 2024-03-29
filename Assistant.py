import threading
import pygame
from OpenAIApiCalls import OpenAPICalls
from OpenAIVisionCalls import OpenVisionCalls

class Assistant:
    def __init__(self):
        self.__api_calls = OpenAPICalls()

    def comprehend_and_response(self):
        AI = self.__api_calls
        user_text = AI.capture_audio()
        response = AI.get_gpt_response(user_text)
        stream = AI.open_ai_tts_stream(response)
        print("User: " + user_text)
        print("AI: " + response)
        self.update_conversational_training_data(user_text, response)
        return user_text, response

    def update_conversational_training_data(self, user_text, response):
        training_data = {
            "messages": [
                {
                    "role": "user",
                    "content": user_text
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }
        with open("./conversation_text/conversational_training_data.txt", "a") as training_data_file:
            training_data_file.write(str(training_data) + '\n')
        with open("./conversation_text/conversation.txt", "a") as conversation_file:
            conversation_file.write(f"Friend:{user_text}\nNora:{response}")

def run_vision_capture():
    vision_calls = OpenVisionCalls()
    vision_calls.capture_frames()

def run_assistant():
    assistant = Assistant()
    stop_condition = False
    while not stop_condition:
        user_text, response = assistant.comprehend_and_response()
        if "go to" and "sleep" in user_text:
            stop_condition = True
            pygame.quit()

if __name__ == "__main__":
    vision_thread = threading.Thread(target=run_vision_capture)
    assistant_thread = threading.Thread(target=run_assistant)

    vision_thread.start()
    assistant_thread.start()

    vision_thread.join()
    assistant_thread.join()
