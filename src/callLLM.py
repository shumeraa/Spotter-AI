from openai import OpenAI
from constants import API_KEY
from pygame import mixer
import time

client = OpenAI(api_key=API_KEY)


def callLLM(prompt):
    # add upon this later
    speak(prompt)


def speak(rep):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=f"Nice job! Completed rep {rep}.",
    ) as response:
        response.stream_to_file(rf"Recordings\output_{rep}.mp3")

    # Initialize Pygame Mixer
    mixer.init()

    # Load the MP3 file
    mixer.music.load(rf"Recordings\output_{rep}.mp3")

    # Play the MP3 file
    mixer.music.play()

    # Wait for the music to finish playing
    while mixer.music.get_busy():
        time.sleep(1)

