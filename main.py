import speech_recognition as sr
from gtts import gTTS
import transformers
import time
import os
import datetime
import numpy as np


class ChatBot:
    """
    :class: ChatBot

    The ChatBot class represents a simple chat bot that can convert speech to text, text to speech, and perform other actions based on specific triggers.

    The ChatBot class has the following methods:

    - `__init__(self, name)`: Initialize the ChatBot object. The `name` parameter sets the name of the chat bot.
    - `speech_to_text(self)`: Convert speech to text using the device's microphone. This method utilizes the SpeechRecognition library and the Google Speech Recognition service. The recognized text is stored in the `self.text` attribute.
    - `text_to_speech(text)`: Convert text to speech using the gTTS library. The `text` parameter specifies the text to be converted. The resulting speech is saved as an audio file and played. The audio file is automatically deleted afterwards.
    - `wake_up(self, text)`: Check if the chat bot's name is mentioned in the given `text`. Returns `True` if the name is found, `False` otherwise.
    - `action_time()`: Get the current time in the format "HH:MM".

    Note:
    - This class requires the following external libraries: SpeechRecognition, gTTS.
    - The `speech_to_text` method includes exception handling for unknown audio and request errors from the Google Speech Recognition service.
    - The `text_to_speech` method calculates the duration of the speech based on the size of the resulting audio file.
    """
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name
        self.text = ""

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
            self.text = "ERROR"
            try:
                self.text = recognizer.recognize_google(audio)
                print("Me  --> ", self.text)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    @staticmethod
    def text_to_speech(text):
        print("Dev --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        os.system('start res.mp3' if os.name == 'nt' else 'afplay res.mp3')
        time.sleep(int(50 * duration))
        os.remove("res.mp3")

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


# Running the AI
if __name__ == "__main__":
    ai = ChatBot(name="dev")
    nlp = transformers.pipeline("text-generation", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ex = True
    while ex:
        ai.speech_to_text()
        if ai.wake_up(ai.text):
            res = "Hello I am Dave the AI, what can I do for you?"
        elif "time" in ai.text:
            res = ai.action_time()
        elif any(i in ai.text for i in ["thank", "thanks"]):
            res = np.random.choice(
                ["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "mention not"])
        elif any(i in ai.text for i in ["exit", "close"]):
            res = np.random.choice(
                ["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "peace out!"])
            ex = False
        else:
            if ai.text == "ERROR":
                res = "Sorry, come again?"
            else:
                chat = nlp(ai.text, max_length=50, pad_token_id=50256)
                res = chat[0]['generated_text'][len(ai.text):].strip()
        ai.text_to_speech(res)
    print("----- Closing down Dev -----")

