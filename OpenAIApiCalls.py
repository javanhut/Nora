import os
import openai
from openai import OpenAI
import dotenv
from SpeechRecognition import SpeechRecognitionWhisper
from AudioVisualizer import AudioVisualizer
from time import time
import pyaudio
import threading
import numpy as np
import queue

class OpenAPICalls(SpeechRecognitionWhisper):

    def __init__(self):
        super().__init__()
        dotenv.load_dotenv()
        self.__client: OpenAI = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.__audio_visualizer = None
        self.audio_playback_queue = queue.Queue()

    @property
    def client(self):
        if not self.__client:
            self.__client = OpenAI(api_key=self.api_key)
        return self.__client

    @property
    def audio_visualizer(self):
        if not self.__audio_visualizer:
            self.__audio_visualizer = AudioVisualizer()
        return self.__audio_visualizer

    def open_ai_tts_stream(self, audio_text: str, print_time: bool = False) -> None:
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 24000
        py_audio_instance = pyaudio.PyAudio()
        player_stream = py_audio_instance.open(
            format=audio_format, channels=channels, rate=rate, output=True
        )
        start_time = time()
        audio_visualizer = self.audio_visualizer

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1", voice="nova", response_format="pcm", input=audio_text
        ) as response:
            if print_time:
                print(f"Time to first byte: {int(time() -start_time) * 1000}ms")
            for chunk in response.iter_bytes(chunk_size=1024):
                player_stream.write(chunk)
                self.audio_playback_queue.put(chunk)
                samples = np.frombuffer(chunk, dtype=np.int16)
                normalized_samples = samples / 32768.0
                audio_visualizer.audio_data_queue.put(normalized_samples)
        if print_time:
            print(f"Done in {int((time() - start_time))*1000}ms")

    def get_gpt_response(self, query: str) -> str:
        chunks = []
        past_conversations = ""
        with open("./conversation_text/conversation.txt", "r") as read_file:
            lines = read_file.readlines()
            for line in lines:
                past_conversations += line
        completion = self.client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:nora1:93WhUaD9",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a emotionally intelligent AI named Nora"
                    f" that builds your personality from past conversations."
                    f" And uses this information as context: {past_conversations}",
                },
                {"role": "user", "content": query},
            ],
            stream=True,
        )

        for chunk in completion:
            chunks.append(chunk.choices[0].delta.content)
        result_string = " ".join(filter(lambda x: x is not None and x != "", chunks))
        return result_string