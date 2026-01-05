import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from langchain.llms.base import LLM
from typing import Optional, List

class SpeechT5TTS(LLM):
    def __init__(self):
        super().__init__()
        # Load model
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Load default speaker voice
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
    
    @property
    def _llm_type(self) -> str:
        return "speecht5_tts"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self.text_to_speech(prompt)
    
    def text_to_speech(self, text: str, output_file: str = "output.wav") -> str:
        """Convert text to speech"""
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"], 
                    self.speaker_embeddings, 
                    vocoder=self.vocoder
                )
            
            # Save audio
            sf.write(output_file, speech.numpy(), samplerate=16000)
            return f"Speech saved to {output_file}"
            
        except Exception as e:
            return f"Error: {str(e)}"


# Usage Example
if __name__ == "__main__":
    # Initialize TTS
    tts = SpeechT5TTS()
    
    # Convert text to speech
    text = "Hello! This is a simple text to speech example."
    result = tts.text_to_speech(text, "hello.wav")
    print(result)
    
    # Using LangChain interface
    response = tts("Welcome to AI speech synthesis!")
    print(response)


# For training with your own voice:
def train_custom_voice(audio_files: List[str], transcripts: List[str]):
    """
    Simple training function for custom voice
    audio_files: List of your voice recordings
    transcripts: List of text transcriptions
    """
    from transformers import Trainer, TrainingArguments
    
    # Load model for training
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./custom_voice",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        save_steps=500,
    )
    
    # Note: You'll need to prepare your dataset here
    # This is a simplified example - actual implementation needs data preprocessing
    
    print("Training setup ready. Prepare your audio files and transcripts.")


# Installation:
# pip install torch transformers datasets soundfile langchain