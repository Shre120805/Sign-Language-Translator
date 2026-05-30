import threading
import queue
import time

try:
    import pyttsx3
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


class SpeechEngine:
    def __init__(self, rate=150, volume=0.9):
        self.enabled = AVAILABLE
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self.rate = rate
        self.volume = volume
        if self.enabled:
            self._start()

    def _start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SPEECH] {e}")

    def speak(self, text):
        if not self.enabled or not text.strip():
            return
        self._queue.put(text)

    def stop(self):
        self._running = False
        if self._queue:
            self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2.0)
