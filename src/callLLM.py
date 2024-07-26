import openai
from dotenv import load_dotenv
import os
from pygame import mixer
import time

load_dotenv()
openai.api_key = os.getenv("API_KEY")


def callLLMs(input_tuple):
    fileName, prompt, rep = input_tuple
    start_time = time.time()
    message = getLLMText(prompt)
    speak(fileName, message, rep)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")


def getLLMText(client_info):
    # Define the new system prompt
    system_prompt = (
        "You are an enthusiastic and knowledgeable personal trainer specializing in improving squat form. "
        "Your goal is to provide clear and concise feedback to help clients perform their squats correctly. "
        "When given information about the client's current rep, "
        "repeat the information in your own words as if you saw it yourself and respond with enthusiasm. "
        "Keep your responses one sentence and very short."
    )

    # Make the API call using the client instance
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": client_info},
        ],
    )

    # Extract the assistant's reply
    return response.choices[0].message.content


def speak(fileName, message, rep):
    fileName = rf"Recordings\{fileName}_{rep}.mp3"

    # if the filename already exists, delete it
    if os.path.exists(fileName):
        os.remove(fileName)

    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=message,
    ) as response:
        response.stream_to_file(fileName)

    # Initialize Pygame Mixer
    mixer.init()

    # Load the MP3 file
    mixer.music.load(fileName)

    # Play the MP3 file
    mixer.music.play()

    # Wait for the music to finish playing
    while mixer.music.get_busy():
        time.sleep(1)
