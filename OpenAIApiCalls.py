import os

import openai
from openai import OpenAI
import dotenv
from SpeechRecognition import SpeechRecognitionWhisper
from pathlib import Path
from time import time
import pyaudio


class OpenAPICalls(SpeechRecognitionWhisper):

    def __init__(self):
        super().__init__()
        dotenv.load_dotenv()
        self.__client: OpenAI = None
        self.api_key = os.getenv("OPENAI_API_KEY")
    @property
    def client(self):
        if not self.__client:
            self.__client = OpenAI(api_key=self.api_key)
        return self.__client

    def open_ai_tts_stream(self, audio_text: str) -> None:
        player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        start_time = time()
        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            response_format="pcm",
            input=audio_text
        ) as response:
            print(f"Time to first byte: {int(time() -start_time) * 1000}ms")
            for chunk in response.iter_bytes(chunk_size=1024):
                player_stream.write(chunk)
        print(f"Done in {int((time() - start_time))*1000}ms")

    def get_gpt_response(self, query: str) -> str:
        chunks = []
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            stream=True
        )

        for chunk in completion:
            chunks.append(chunk.choices[0].delta.content)
        input_list = chunks
        result_string = ' '.join(filter(lambda x: x is not None and x != '', input_list))
        return result_string
