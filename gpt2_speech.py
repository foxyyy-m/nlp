!pip install transformers
!pip install pyaudio
!pip install SpeechRecognition
!pip install gTTS
!pip install playsound

import torch
import transformers
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound


model_stt = transformers.Wav2Vec2ForCTC.from_pretrained('')
model_tts = transformers.Wav2Vec2ForCTC.from_pretrained('')

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

text = r.recognize_google(audio)
kenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')


input_ids = kenizer.encode(text, return_tensors='pt')

output = model.generate(input_ids=input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
response = tokenizer.decode(output[0], skip_special_tokens=True)
speech_output = gTTS(response)


speech_output.save('response.mp3')
playsound('response.mp3')
