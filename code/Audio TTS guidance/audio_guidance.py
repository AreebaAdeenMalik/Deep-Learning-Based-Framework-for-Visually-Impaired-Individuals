"""
Audio Guidance System Module
=============================
Provides real-time text-to-speech (TTS) guidance for visually impaired navigation.
Handles non-blocking audio output with priority-based message queueing.

"""

import threading
import queue
import time

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[Audio] WARNING: pyttsx3 not installed. Audio guidance will be disabled.")
    print("[Audio] Install with: pip install pyttsx3")


class AudioGuidanceSystem:
    """
    Text-to-speech based guidance system for real-time assistive navigation.
    Manages message queue and non-blocking audio output.
    """

    def __init__(self, rate=150, volume=1.0, voice_id=0):
        """
        Initialize audio guidance system.

        Args:
            rate (int): Speech rate (words per minute, default 150)
            volume (float): Volume level (0.0-1.0, default 1.0)
            voice_id (int): Voice index (0=male, 1=female if available)
        """
        if not PYTTSX3_AVAILABLE:
            print("[Audio] pyttsx3 not available - audio disabled")
            self.engine = None
            self.audio_disabled = True
            return

        self.audio_disabled = False
        self.engine = pyttsx3.init()
        
        # Configure speech parameters
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voice
        voices = self.engine.getProperty('voices')
        if voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)
        
        # Message queue for non-blocking operation
        self.message_queue = queue.Queue()
        self.is_running = True
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        
        print("[Audio] Guidance system initialized")

    def _audio_worker(self):
        """
        Worker thread that processes messages from queue.
        Runs in background to avoid blocking main thread.
        """
        if self.audio_disabled or not self.engine:
            return

        while self.is_running:
            try:
                # Get message from queue with timeout
                message, priority = self.message_queue.get(timeout=0.5)
                
                # Speak the message
                self.engine.say(message)
                self.engine.runAndWait()
                
                self.message_queue.task_done()
                
            except queue.Empty:
                # No message in queue, continue waiting
                pass
            except Exception as e:
                print(f"[Audio] Error in worker thread: {str(e)}")

    def speak(self, message, priority=1, wait=False):
        """
        Queue a message for text-to-speech.

        Args:
            message (str): Message to speak
            priority (int): Priority level (higher = urgent, 0-10)
            wait (bool): If True, block until message is spoken
        """
        if self.audio_disabled or not self.engine:
            return

        if not message or not isinstance(message, str):
            return
        
        try:
            self.message_queue.put((message, priority), block=False)
            
            if wait:
                self.message_queue.join()  # Wait for queue to empty
                
        except queue.Full:
            print(f"[Audio] Message queue full, dropping: {message}")

    def get_queue_size(self):
        """Get number of pending messages in queue."""
        if self.audio_disabled:
            return 0
        return self.message_queue.qsize()

    def clear_queue(self):
        """Clear all pending messages from queue."""
        if self.audio_disabled:
            return
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Shutdown audio system gracefully."""
        if self.audio_disabled:
            return
        print("[Audio] Shutting down...")
        self.is_running = False
        self.clear_queue()
        self.audio_thread.join(timeout=2.0)
        print("[Audio] Shutdown complete")


class SafetyAlerts:
    """
    Pre-configured safety alert messages for assistive navigation.
    """

    # Obstacle warnings
    OBSTACLE_AHEAD = "Obstacle ahead"
    OBSTACLE_LEFT = "Obstacle on the left"
    OBSTACLE_RIGHT = "Obstacle on the right"
    CLEAR_PATH = "Path is clear"
    
    # Distance-based warnings
    @staticmethod
    def obstacle_distance(distance_m):
        """Generate distance-based warning."""
        if distance_m < 0.5:
            return f"Very close obstacle, {distance_m:.1f} meters away"
        elif distance_m < 1.0:
            return f"Close obstacle, {distance_m:.1f} meters away"
        elif distance_m < 2.0:
            return f"Obstacle {distance_m:.1f} meters away"
        else:
            return f"Object detected {distance_m:.1f} meters ahead"

    # Object-specific warnings
    @staticmethod
    def object_detected(obj_class, distance_m=None, direction=None):
        """Generate object detection message."""
        msg = f"{obj_class} detected"
        
        if direction:
            msg += f" on the {direction}"
        if distance_m:
            msg += f" {distance_m:.1f} meters"
        
        return msg

    # Navigation guidance
    MOVE_LEFT = "Move slightly to the left"
    MOVE_RIGHT = "Move slightly to the right"
    MOVE_FORWARD = "Move forward carefully"
    STOP = "Stop immediately"
    
    # Door and obstacle detection
    @staticmethod
    def obstacle_type(obj_class):
        """Get appropriate message for obstacle type."""
        messages = {
            'person': 'Person detected, move aside',
            'door': 'Door detected ahead',
            'stairs': 'Stairs detected, be careful',
            'chair': 'Chair detected',
            'table': 'Table detected',
            'car': 'Vehicle detected',
            'bicycle': 'Bicycle detected',
        }
        return messages.get(obj_class, f'{obj_class} detected')

    # Status messages
    SYSTEM_READY = "System ready for navigation"
    SYSTEM_INACTIVE = "Navigation system inactive"
    CALIBRATION_REQUIRED = "Camera calibration required"
    GPS_SIGNAL_LOST = "GPS signal lost"


def demo_audio_system():
    """
    Demonstration of the audio guidance system.
    """
    print("\n" + "="*60)
    print("Audio Guidance System Demo")
    print("="*60)
    
    # Initialize system
    audio = AudioGuidanceSystem(rate=150, volume=1.0)
    
    # Test messages
    test_messages = [
        (SafetyAlerts.SYSTEM_READY, 5),
        (SafetyAlerts.obstacle_detected('person', 2.5, 'left'), 4),
        (SafetyAlerts.obstacle_distance(1.2), 5),
        (SafetyAlerts.MOVE_RIGHT, 4),
        (SafetyAlerts.CLEAR_PATH, 3),
    ]
    
    # Queue all messages
    for msg, priority in test_messages:
        print(f"Queuing: {msg}")
        audio.speak(msg, priority=priority)
        time.sleep(0.5)
    
    # Wait for all messages to be spoken
    print("\nWaiting for audio to complete...")
    audio.message_queue.join()
    
    # Shutdown
    audio.shutdown()
    print("Demo complete!")


if __name__ == "__main__":
    demo_audio_system()




