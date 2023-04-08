!pip install openai
!pip install pyaudio
!pip install SpeechRecognition
!pip install gTTS
!pip install playsound

import openai
import os
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound


openai.api_key = os.environ["key"]
r = sr.Recognizer()

mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

text = r.recognize_google(audio)
prompt = text.strip()
response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=50,
  n=1,
  stop=None,
  temperature=0.5,
)

response_text = response.choices[0].text.strip()
speech_output = gTTS(response_text)
speech_output.save('response.mp3')
playsound('response.mp3')
