<div style="display: flex; gap: 10px;">
  <img src="https://github.com/shumeraa/Spotter-AI/blob/main/gifs/normalSquat.gif" width="250" height="250"/>
  <img src="https://github.com/shumeraa/Spotter-AI/blob/main/gifs/shortSquat.gif" width="250" height="250"/>
  <img src="https://github.com/shumeraa/Spotter-AI/blob/main/gifs/collapseSquat.gif" width="250" height="250"/>
</div>



# Welcome to Spotter-AI!
Spotter-AI is your personal trainer for real-time squat form feedback! Leveraging YOLOv8 for pose estimation, Spotter-AI assesses your squat technique and relays the data to GPT-4.0-mini. GPT crafts a clear, user-friendly message, which is then vocalized by OpenAI TTS, guiding you through your workout.

Spotter-AI currently monitors three key aspects of your squat: rep count, squat depth, and knee alignment. Check out the GIFs above to see it in action!

# How to Start:
- Create a .env file in src and add your openAI api key as API_KEY
- Create a Recordings folder and set it in main.py
- Create a Data folder and add your squat video

1. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   ```
2. **Activate the Virtual Environment:**
    - On Windows:
    ```bash
    .venv\Scripts\activate
    ```
    - On Linux:
    ```bash
    source .venv/bin/activate
    ```
3. **Install the Required Packages:**

    ```bash
    pip install -r requirements.txt
    ```


And then run main.py!
