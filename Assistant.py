import threading

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
        user_text = AI.capture_audio()  # This needs to be a blocking call
        response = AI.get_gpt_response(user_text)
        stream = AI.open_ai_tts_stream(response)
        return user_text, stream

    def comprehend_and_response(self):
        user_text, stream = self.capture_and_respond()
        # thread1 = threading.Thread(target=lambda: print(user_text))
        thread2 = threading.Thread(target=lambda: print(stream))
        # thread1.start()
        thread2.start()
        # thread1.join()
        thread2.join()
        return user_text


if "__main__" == __name__:
    stop_condition = False
    while not stop_condition:
        assistant = Assistant()
        user_text = assistant.comprehend_and_response()
        if "stop" in user_text:
            stop_condition = True