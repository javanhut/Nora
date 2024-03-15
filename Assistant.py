import sys
import threading
import pygame
from OpenAIApiCalls import OpenAPICalls


class Assistant:

    def __init__(self):
        self.__api_calls = None

    @property
    def api(self):
        if not self.__api_calls:
            self.__api_calls = OpenAPICalls()
        return self.__api_calls

    def capture_and_respond(self):
        AI = self.api
        user_text = AI.capture_audio()
        response = AI.get_gpt_response(user_text)
        stream = AI.open_ai_tts_stream(response)
        with open("./conversation_text/conversation.txt", "a") as conversation_file:
            conversation_file.write(f"Friend: {user_text} \n")
            conversation_file.write(response + "\n")
        thread1 = threading.Thread(target=AI)
        thread1.start()
        thread1.join()
        return user_text, response

    def comprehend_and_response(self):
        user_text, response = self.capture_and_respond()
        print("User: " + user_text)
        print("AI: " + response)

        return user_text


if "__main__" == __name__:
    stop_condition = False
    while not stop_condition:
        assistant = Assistant()
        user_text = assistant.comprehend_and_response()
        if "stop" in user_text:
            stop_condition = True
            pygame.quit()
