from openai import OpenAI
from constants import API_KEY

client = OpenAI(api_key=API_KEY)

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="onyx",
    input="Hello world! This is a streaming test.",
) as response:
    response.stream_to_file(r"Recordings\output.mp3")
