# Method 1: Using SpeechT5 from Hugging Face
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import librosa
import numpy as np

def clone_voice_speecht5(reference_audio_path, text_to_speak, output_path="speecht5_output.wav"):
    """
    Voice cloning using SpeechT5 from Hugging Face
    """
    # Load models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    # Load and process reference audio
    audio, sr = librosa.load(reference_audio_path, sr=16000)
    
    # Create speaker embeddings from reference audio
    # Simple approach - use mean of audio features as speaker embedding
    speaker_embeddings = torch.tensor(audio).mean(dim=0, keepdim=True).unsqueeze(0)
    speaker_embeddings = speaker_embeddings.repeat(1, 512)  # SpeechT5 expects 512-dim embeddings
    
    # Process text
    inputs = processor(text=text_to_speak, return_tensors="pt")
    
    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Save output
    sf.write(output_path, speech.numpy(), samplerate=16000)
    print(f"SpeechT5 output saved to: {output_path}")

# Usage examples
if __name__ == "__main__":
    reference_file = "voice/kejriwal.mp4"
    text = "Hello, this is a test of voice cloning."
    
    # Method 1: SpeechT5 (Recommended - Good balance)
    try:
        clone_voice_speecht5(reference_file, text, "speecht5_output.wav")
    except Exception as e:
        print(f"SpeechT5 failed: {e}")


# Installation commands for different methods:
"""
# For SpeechT5:
pip install transformers datasets soundfile librosa

# For Bark:
pip install git+https://github.com/suno-ai/bark.git

# For SV2TTS:
git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
# Follow their installation instructions

# For Fairseq:
pip install fairseq
"""