import speech_recognition as sr
import pyttsx3
import threading
import time
import queue
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAIAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

        # Configure TTS
        self.setup_tts()

        # Control flags
        self.listening = True
        self.processing = False

        # Queue for streaming text
        self.speech_queue = queue.Queue()

        # Adjust for ambient noise
        with self.microphone as source:
            logger.info("Adjusting for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            logger.info("Ready to listen!")

    def setup_tts(self):
        """Configure text-to-speech engine"""
        voices = self.tts_engine.getProperty("voices")
        # Try to set a female voice if available
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                self.tts_engine.setProperty("voice", voice.id)
                break

        # Adjust speech rate and volume
        self.tts_engine.setProperty("rate", 180)  # Speed of speech
        self.tts_engine.setProperty("volume", 0.9)  # Volume level (0.0 to 1.0)

    def listen_for_wake_word(self):
        """Continuously listen for the wake word 'hey alexa'"""
        logger.info("Listening for 'Hey Alexa'...")

        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source, timeout=1, phrase_time_limit=3
                    )

                try:
                    # Recognize speech
                    text = self.recognizer.recognize_google(audio).lower()
                    logger.info(f"Heard: {text}")

                    # Check for wake word
                    if "hey alexa" in text or "hey alex" in text:
                        logger.info("Wake word detected!")
                        self.speak("Yes, how can I help you?")
                        self.listen_for_query()

                except sr.UnknownValueError:
                    # Ignore unrecognized speech
                    pass
                except sr.RequestError as e:
                    logger.error(
                        f"Could not request results from speech recognition service: {e}"
                    )
                    time.sleep(1)

            except sr.WaitTimeoutError:
                # Timeout is normal, continue listening
                pass
            except Exception as e:
                logger.error(f"Error in wake word detection: {e}")
                time.sleep(1)

    def listen_for_query(self):
        """Listen for the user's query after wake word is detected"""
        if self.processing:
            return

        self.processing = True
        logger.info("Listening for your query...")

        try:
            with self.microphone as source:
                # Give user time to speak
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

            try:
                query = self.recognizer.recognize_google(audio)
                logger.info(f"Query: {query}")

                # Process query with Gemini
                self.process_query(query)

            except sr.UnknownValueError:
                self.speak(
                    "I'm sorry, I didn't understand that. Could you please repeat?"
                )
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                self.speak("Sorry, I'm having trouble with speech recognition.")

        except sr.WaitTimeoutError:
            self.speak("I didn't hear anything. Try saying 'Hey Alexa' again.")
        except Exception as e:
            logger.error(f"Error listening for query: {e}")
            self.speak("Sorry, there was an error processing your request.")

        finally:
            self.processing = False

    def process_query(self, query):
        """Process the query using Gemini API with streaming"""
        logger.info(f"Processing query: {query}")

        try:
            # Add context to make responses more conversational
            formatted_query = (
                f"Please provide a helpful and conversational response to: {query}"
            )

            # Stream the response
            self.speak("Let me think about that...")

            # Start TTS thread
            tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
            tts_thread.start()

            # Stream response from Gemini
            response_text = ""
            for chunk in self.model.stream(formatted_query):
                if hasattr(chunk, "content") and chunk.content:
                    response_text += chunk.content
                    # Add chunks to speech queue for streaming
                    self.speech_queue.put(chunk.content)

            # Signal end of streaming
            self.speech_queue.put(None)

            logger.info(f"Response: {response_text}")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            self.speak("Sorry, I encountered an error while processing your request.")

    def tts_worker(self):
        """Worker thread for text-to-speech streaming"""
        buffer = ""

        while True:
            try:
                chunk = self.speech_queue.get(timeout=1)
                if chunk is None:  # End of stream signal
                    if buffer.strip():  # Speak any remaining text
                        self.speak_text(buffer)
                    break

                buffer += chunk

                # Speak when we have a complete sentence or enough text
                if (
                    any(punct in buffer for punct in [". ", "! ", "? ", "\n"])
                    or len(buffer) > 100
                ):
                    # Find the last complete sentence
                    last_punct = max(
                        buffer.rfind(". "), buffer.rfind("! "), buffer.rfind("? ")
                    )

                    if last_punct > 0:
                        to_speak = buffer[: last_punct + 1].strip()
                        buffer = buffer[last_punct + 1 :].strip()

                        if to_speak:
                            self.speak_text(to_speak)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS worker: {e}")
                break

    def speak_text(self, text):
        """Speak the given text"""
        if text.strip():
            logger.info(f"Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def speak(self, text):
        """Simple speak method for system messages"""
        logger.info(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def stop(self):
        """Stop the assistant"""
        self.listening = False
        logger.info("Assistant stopped.")

    def run(self):
        """Main method to run the assistant"""
        logger.info("Starting Voice AI Assistant...")
        self.speak("Voice assistant is ready. Say 'Hey Alexa' to activate me.")

        try:
            # Start listening in main thread
            self.listen_for_wake_word()
        except KeyboardInterrupt:
            logger.info("Stopping assistant...")
            self.speak("Goodbye!")
            self.stop()


def main():
    """Main function to run the assistant"""
    assistant = VoiceAIAssistant()

    try:
        assistant.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("Please make sure you have:")
        print("1. Set up your GOOGLE_API_KEY in .env file")
        print(
            "2. Installed required packages: pip install speechrecognition pyttsx3 pyaudio langchain-google-genai python-dotenv"
        )
        print("3. Connected a microphone")


if __name__ == "__main__":
    main()
