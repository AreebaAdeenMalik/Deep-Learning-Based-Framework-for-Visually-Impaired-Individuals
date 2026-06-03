"""
Audio Guidance System
=====================
Pure TTS player. The previous rule-based `SafetyAlerts` catalogue (e.g.
"Move slightly to the left", "Stairs detected, be careful", etc.) has
been REMOVED - LLaVA now generates every spoken sentence based on the
fused multimodal context.

The only rule-based string that remains here is `EMERGENCY_STOP`, which
is uttered when the fusion engine raises the latency-critical emergency
flag. This is intentional: emergency stops must never wait on a 1-3 s
LLM call (per the thesis spec).
"""

import queue
import threading
import time

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[Audio] WARNING: pyttsx3 not installed. Install with: pip install pyttsx3")


# Generic fallback used only when the fusion engine cannot describe the
# obstacle (e.g. emergency_reason is empty). When the safety layer DOES
# describe it, `build_emergency_message()` constructs a data-driven line
# using the actual object class and side from the fused context.
EMERGENCY_STOP_MESSAGE = "Stop. Obstacle directly ahead."


def build_emergency_message(
    label: str | None = None,
    side: str | None = None,
    proximity_label: str | None = None,
) -> str:
    """Build a short data-driven emergency utterance.

    Examples:
        ('person', 'center', 'near')  -> "Stop. Person very close ahead."
        ('car', 'left', 'near')       -> "Stop. Car very close on the left."
        (None, None, None)            -> EMERGENCY_STOP_MESSAGE
    """
    if not label:
        return EMERGENCY_STOP_MESSAGE

    # Map proximity_label to a spoken qualifier.
    qualifier = {
        "near": "very close",
        "mid": "close",
        "far": "ahead",
    }.get(proximity_label or "", "close")

    # Map horizontal zone to natural English direction.
    direction = {
        "center": "ahead",
        "left": "on the left",
        "right": "on the right",
    }.get(side or "center", "ahead")

    return f"Stop. {label.capitalize()} {qualifier} {direction}."


class AudioGuidanceSystem:
    """Non-blocking text-to-speech player with priority interrupt."""

    def __init__(self, rate=160, volume=1.0, voice_id=0):
        if not PYTTSX3_AVAILABLE:
            self.engine = None
            self.audio_disabled = True
            print("[Audio] pyttsx3 unavailable - audio disabled")
            return

        self.audio_disabled = False
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        voices = self.engine.getProperty("voices")
        if voices and voice_id < len(voices):
            self.engine.setProperty("voice", voices[voice_id].id)

        self.message_queue: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = 0  # tie-breaker to keep PriorityQueue stable
        self.is_running = True

        self._last_spoken = ""
        self._last_spoken_at = 0.0
        self._dedup_window_s = 2.0  # don't repeat the same sentence within 2s

        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        print("[Audio] Guidance system initialized")

    # ----------------------- worker ----------------------- #

    def _audio_worker(self):
        if self.audio_disabled or not self.engine:
            return
        while self.is_running:
            try:
                priority, _seq, message = self.message_queue.get(timeout=0.5)
                if message is None:
                    self.message_queue.task_done()
                    continue
                self.engine.say(message)
                self.engine.runAndWait()
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Audio] Worker error: {e}")

    # ----------------------- public API ----------------------- #

    def speak(self, message, priority=5, wait=False):
        """Queue an utterance. Lower `priority` value = spoken sooner."""
        if self.audio_disabled or not self.engine:
            return
        if not isinstance(message, str) or not message.strip():
            return

        now = time.time()
        # Suppress duplicate repeats inside a short window (very common with
        # nearly-identical LLaVA outputs across frames).
        if message == self._last_spoken and (now - self._last_spoken_at) < self._dedup_window_s:
            return
        self._last_spoken = message
        self._last_spoken_at = now

        self._seq += 1
        try:
            self.message_queue.put((priority, self._seq, message), block=False)
            if wait:
                self.message_queue.join()
        except queue.Full:
            print(f"[Audio] queue full - dropped: {message}")

    def emergency_speak(self, message: str = EMERGENCY_STOP_MESSAGE):
        """Highest priority. Clears the queue first to interrupt anything in flight."""
        if self.audio_disabled or not self.engine:
            return
        self.clear_queue()
        self._seq += 1
        try:
            self.message_queue.put((0, self._seq, message), block=False)
        except queue.Full:
            pass

    def clear_queue(self):
        if self.audio_disabled:
            return
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except queue.Empty:
                break

    def get_queue_size(self):
        if self.audio_disabled:
            return 0
        return self.message_queue.qsize()

    def shutdown(self):
        if self.audio_disabled:
            return
        print("[Audio] Shutting down...")
        self.is_running = False
        self.clear_queue()
        self.audio_thread.join(timeout=2.0)
        print("[Audio] Shutdown complete")


if __name__ == "__main__":
    a = AudioGuidanceSystem()
    a.speak("System ready.", priority=3)
    a.speak("Path looks clear. Continue forward.", priority=5)
    a.emergency_speak()
    time.sleep(6)
    a.shutdown()
