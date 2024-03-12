import threading
import time

from OpenAIApiCalls import OpenAPICalls


class Assistant:

    def __init__(self):
        self.__api_calls: OpenAPICalls = None
    @property
    def api(self):
        if not self.__api_calls:
            self.__api_calls = OpenAPICalls()
            return self.__api_calls

    def comprehend_and_response(self):
        AI = self.api
        user_text = AI.capture_audio()
        time.sleep(1)
        AI.open_ai_tts_stream(AI.get_gpt_response(user_text))
        # return user_text

    def start_conversation(self):
        thread = threading.Thread(target=self.comprehend_and_response)
        thread.start()



if "__main__" == __name__:
    nova_assistant = Assistant()
    nova_assistant.start_conversation()