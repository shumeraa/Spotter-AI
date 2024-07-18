from multiprocessing import Process, Event
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
from pygame import mixer


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print(event.event_type, event.src_path)
        print("SOMETHING HAPPENED")

    def on_created(self, event):
        print("on_created", event.src_path)
        if event.src_path.lower().endswith(".mp3"):
            print("MP3 file detected:", event.src_path)
            time.sleep(2)  # Wait a second to ensure the file is fully written
            mixer.init()
            try:
                mixer.music.load(os.path.abspath(event.src_path))
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)
            except pygame.error as e:
                print(f"Failed to load or play the file: {e}")


observer = None
stop_event = Event()  # Event to control the stopping of the loop


def start_monitoring(path="Recordings"):
    global observer
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()

    while not stop_event.is_set():
        time.sleep(1)  # Replaces the infinite loop with a stoppable loop

    observer.stop()
    observer.join()


def stop_monitoring():
    global observer
    stop_event.set()  # Signal the loop to stop
    print("Monitoring stopped.")
