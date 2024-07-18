from openai import OpenAI
from constants import API_KEY

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
