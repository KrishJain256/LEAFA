import pyttsx3
import os
from decouple import config
import speech_recognition as sr
from random import choice
from utils import opening_text
from datetime import datetime

USERNAME = config('USER')
BOTNAME = config('BOTNAME')


# engine = pyttsx3.init('_espeak')
#
# # Set Rate
# engine.setProperty('rate', 190)
#
# # Set Volume
# engine.setProperty('volume', 1.0)
#
# # Set Voice (Female)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)


# Text to Speech Conversion
def speak(text):
    """Used to speak whatever text is passed to it"""
    return os.system("espeak -vmb-us1+f5 -s 110 -a 100 -p 60 \"" + text + "\" ")
    # engine.say(text)
    # engine.runAndWait()

def take_user_input():
    """Takes user input, recognizes it using Speech Recognition module and converts it into text"""

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening....')
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print('Recognizing...')
        query = r.recognize_google(audio, language='en-in')
        if not 'exit' in query or 'stop' in query:
            speak(choice(opening_text))
        else:
            hour = datetime.now().hour
            if hour >= 21 and hour < 6:
                speak("Good night sir, take care!")
            else:
                speak('Have a good day sir!')
            exit()
    except Exception:
        speak('Sorry, I could not understand. Could you please say that again?')
        query = 'None'
    return query

print(take_user_input())