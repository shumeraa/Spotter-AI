![](https://github.com/shumeraa/Spotter-AI/blob/main/gifs/normalSquat.gif)
![](https://github.com/shumeraa/Spotter-AI/blob/main/gifs/shortSquat.gif)
![](https://github.com/shumeraa/Spotter-AI/blob/main/gifs/collapseSquat.gif)
# Welcome to Spotter-AI!
Spotter-AI is your personal trainer for real-time squat form feedback! Leveraging YOLOv8 for pose estimation, Spotter-AI assesses your squat technique and relays the data to GPT-4.0-mini. GPT crafts a clear, user-friendly message, which is then vocalized by OpenAI TTS, guiding you through your workout.

Spotter-AI currently monitors three key aspects of your squat: rep count, squat depth, and knee alignment. Check out the GIFs above to see it in action!

# How to Start:
- Create a .env file in src and add your openAI api key as API_KEY
- Create a Recordings folder and set it in main.py
- Create a Data folder and add your squat video

Create and activate your venv:

```
python -m venv .venv
```

```
.venv\Scripts\activate
```

And then run main.py!
