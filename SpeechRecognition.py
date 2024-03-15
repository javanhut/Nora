import speech_recognition as sr
from speech_recognition import Microphone, Recognizer


class SpeechRecognitionWhisper:
    """This module turn speech into text using Whisper API from Speech Recognition."""

    def __init__(self):
        self.__microphone: Microphone = None
        self.__recognizer: Recognizer = None

    @property
    def Microphone(self) -> Microphone:
        """This returns a microphone object."""
        if not self.__microphone:
            self.__microphone = Microphone()
        return self.__microphone

    @property
    def Recognizer(self) -> Recognizer:
        """This returns a recognizer object."""
        if not self.__recognizer:
            self.__recognizer = Recognizer()
        return self.__recognizer

    def capture_audio(self) -> str:
        """This method captures audio from the microphone and returns a string."""
        with Microphone() as source:
            Recognizer().adjust_for_ambient_noise(source)
            print("Capturing audio....")
            audio = Recognizer().listen(source)
        try:
            audio_text = Recognizer().recognize_whisper(audio, language="english")
            return audio_text
        except sr.UnknownValueError:
            print("Whisper couldn't understand audio ")
        except sr.RequestError as e:
            print("Could not request results from Whisper")


# s1 = SpeechRecognitionWhisper()
# print(s1.capture_audio())
