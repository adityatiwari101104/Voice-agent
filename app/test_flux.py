import os
import threading
import pyaudio

from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType

load_dotenv()

api_key = os.getenv("DEEPGRAM_API_KEY")

if not api_key:
    raise RuntimeError("DEEPGRAM_API_KEY not found in .env")

client = DeepgramClient(api_key=api_key)

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=2560
)


def on_open(message):
    print("✅ Connected to Deepgram Flux")


def on_message(message):
    message_type = getattr(message, "type", None)

    if message_type == "Connected":
        print("✅ Flux connection ready")
        return

    if message_type == "TurnInfo":
        event = getattr(message, "event", None)
        transcript = getattr(message, "transcript", None)
        turn_index = getattr(message, "turn_index", None)
        confidence = getattr(
            message,
            "end_of_turn_confidence",
            None
        )

        if transcript:
            print(f"[{event}] {transcript}")

        if event == "StartOfTurn":
            print(f"\n--- StartOfTurn: {turn_index} ---")

        elif event == "EndOfTurn":
            print(
                f"\n--- EndOfTurn: {turn_index} "
                f"| confidence={confidence} ---"
            )


def on_error(error):
    print(f"\n❌ Deepgram error: {error}")


try:
    with client.listen.v2.connect(
        model="flux-general-multi",
        eot_threshold=0.7,
        eot_timeout_ms=3000,
        encoding="linear16",
        sample_rate=16000,
    ) as connection:

        connection.on(EventType.OPEN, on_open)
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        listener_thread = threading.Thread(
            target=connection.start_listening,
            daemon=True
        )

        listener_thread.start()

        print("🎙️ Speak now.")
        print("Stop speaking and wait ~3 seconds.")
        print("Press Ctrl+C only to exit.\n")

        while True:
            data = stream.read(
                2560,
                exception_on_overflow=False
            )

            connection.send_media(data)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()