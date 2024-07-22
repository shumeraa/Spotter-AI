import openai
from constants import API_KEY
from pygame import mixer
import time

# client = OpenAI(api_key=API_KEY)
openai.api_key = API_KEY


def callLLMs(input_tuple):
    prompt, rep = input_tuple
    start_time = time.time()
    message = getLLMText(prompt)
    speak(message, rep)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")


def getLLMText(client_info):
    # Define the new system prompt
    system_prompt = (
        "You are an enthusiastic and knowledgeable personal trainer specializing in improving squat form. "
        "Your goal is to provide clear and concise feedback to help clients perform their squats correctly. "
        "When given information about the client's current rep, "
        "respond with enthusiasm and repeat the information in your own words. "
        "Keep your responses motivational and to the point."
    )

    # Make the API call using the client instance
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": client_info},
        ],
    )

    # Extract the assistant's reply
    return response.choices[0].message.content


def speak(message, rep):
    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=message,
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

